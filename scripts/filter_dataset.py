"""Embedding purity gate for F5-TTS training clips.

Word-level chunking (build_finetune_dataset.py) keeps the other speaker's TURNS
out of a clip, but cannot see simultaneous overlap: when both people talk at
once the diarizer still stamps the word with the target's label, and the audio
slice contains both voices. This gate catches that acoustically: it builds a
wespeaker voice centroid per diarized speaker from the source recording, then
two complementary detectors (calibrated on listened labels; each catches a
contamination mode the other is blind to):

1. Identity margin (wespeaker, catches SEQUENTIAL male speech, e.g. a male word
   at a clip edge): full-2s windows, last anchored to the clip end (sub-2s tail
   windows embed noisily). A window fails only on POSITIVE evidence of another
   speaker: their sim >= --other-min while the target's lead is < --margin. No
   absolute floor -- laughter/expressive delivery scores low vs BOTH centroids
   and must not be rejected. Clip-level backstop: mean target sim >=
   --clip-mean-min drops clips that are entirely off-voice.
2. Overlapped-speech detection (pyannote segmentation-3.0, catches SIMULTANEOUS
   speech incl. singing/vocalizing, which barely moves identity similarity):
   reject when frames with >=2 active speakers total > --max-overlap seconds.
   Labeled data: pure clips <=0.19s, contaminated >=0.93s.

Rejected wavs move to <wav-dir>\..\rejected\ for audit and their rows are
removed from metadata.csv.

Runs in the WhisperX venv (pyannote.audio). Uses GPU if free (the training GPU
is idle at this point in the pipeline), CPU otherwise.
"""
import argparse
import json
import os
import shutil
import sys
import wave

import numpy as np
import torch


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def wav_duration(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="diarization JSON of the source recording")
    ap.add_argument("--audio", required=True, help="source recording wav (for centroids)")
    ap.add_argument("--speaker", required=True, help="target speaker index in this recording")
    ap.add_argument("--wav-dir", required=True, help="dataset wavs/ dir")
    ap.add_argument("--meta", required=True, help="metadata.csv to prune")
    ap.add_argument("--start-index", type=int, default=0,
                    help="only gate seg_NNNNN.wav with NNNNN >= this (the clips this source produced)")
    ap.add_argument("--other-min", type=float, default=0.45,
                    help="other-speaker sim floor: below this a window can't be rejected (no evidence)")
    ap.add_argument("--margin", type=float, default=0.20,
                    help="min target lead over the other speaker when other-min is reached "
                         "(0.20 calibrated on listened labels: contaminated clips scored <=0.197, "
                         "pure >=0.208 on anchored 2s windows)")
    ap.add_argument("--clip-mean-min", type=float, default=0.30,
                    help="min mean target sim across a clip's windows (drops fully off-voice clips)")
    ap.add_argument("--max-overlap", type=float, default=0.30,
                    help="max seconds of detected overlapped speech per clip "
                         "(labeled pure clips <=0.19s, contaminated >=0.93s)")
    args = ap.parse_args()

    target = f"SPEAKER_{int(args.speaker):02d}"
    with open(args.json, encoding="utf-8") as f:
        segs = json.load(f)["segments"]

    from pyannote.audio import Inference, Model
    from pyannote.core import Segment

    tok = os.environ.get("HF_TOKEN")
    try:
        model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", use_auth_token=tok)
    except TypeError:
        model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", token=tok)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inf = Inference(model, window="whole", device=device)
    try:
        seg_model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=tok)
    except TypeError:
        seg_model = Model.from_pretrained("pyannote/segmentation-3.0", token=tok)
    seg_inf = Inference(seg_model, device=device)
    eprint(f"Purity gate on {device.type}: target {target}, "
           f"reject if other>={args.other_min} & margin<{args.margin}, "
           f"clip mean>={args.clip_mean_min}, overlap<={args.max_overlap}s")

    def emb(audio_file, a, b):
        v = np.asarray(inf.crop(audio_file, Segment(a, b))).reshape(-1)
        return v / (np.linalg.norm(v) + 1e-9)

    # Voice centroid per diarized speaker, from that speaker's longest segments.
    centroids = {}
    for lab in sorted({s.get("speaker") for s in segs if s.get("speaker")}):
        spans = [(s["start"], s["end"]) for s in segs
                 if s.get("speaker") == lab and "start" in s and 3.0 <= s["end"] - s["start"] <= 15.0]
        spans.sort(key=lambda x: -(x[1] - x[0]))
        vecs = [emb(args.audio, a, b) for a, b in spans[:8]]
        if vecs:
            c = np.mean(vecs, axis=0)
            centroids[lab] = c / (np.linalg.norm(c) + 1e-9)
    if target not in centroids or len(centroids) < 2:
        eprint(f"WARN: need target + other centroids (have {sorted(centroids)}); gate skipped.")
        return
    tgt_c = centroids[target]
    others = [c for lab, c in centroids.items() if lab != target]

    def overlap_seconds(path, dur, thresh=0.5):
        # Multilabel speaker activities per 10s chunk (chunks x frames x 3);
        # max-aggregate chunk overlaps, count frames with >=2 active speakers.
        out = seg_inf(path)
        data = out.data if out.data.ndim == 3 else out.data[None]
        n_frames = data.shape[1]
        frame_dur = out.sliding_window.duration / n_frames
        total = int(np.ceil(dur / frame_dur)) + 1
        act = np.zeros((total, data.shape[2]))
        for c in range(data.shape[0]):
            off = int(round(c * out.sliding_window.step / frame_dur))
            end = min(off + n_frames, total)
            act[off:end] = np.maximum(act[off:end], data[c][:end - off])
        return float((((act > thresh).sum(axis=1)) >= 2).sum() * frame_dur)

    def clip_ok(path):
        dur = wav_duration(path) - 0.01   # epsilon: crop at exact EOF raises in pyannote
        # Detector 1: identity margin on full-2s windows (last anchored to clip end).
        starts = list(np.arange(0.0, max(dur - 2.0, 0.0), 2.0)) + [max(dur - 2.0, 0.0)]
        sims = []
        for t in starts:
            e = emb(path, t, min(t + 2.0, dur))
            sim_t = float(np.dot(e, tgt_c))
            sim_o = max(float(np.dot(e, c)) for c in others)
            sims.append(sim_t)
            if sim_o >= args.other_min and sim_t - sim_o < args.margin:
                return False, f"@{t:.0f}s other present (target {sim_t:.2f}, other {sim_o:.2f})"
        if float(np.mean(sims)) < args.clip_mean_min:
            return False, f"off-voice (mean target {float(np.mean(sims)):.2f})"
        # Detector 2: simultaneous overlapped speech (incl. singing).
        ov = overlap_seconds(path, dur)
        if ov > args.max_overlap:
            return False, f"overlapped speech {ov:.1f}s"
        return True, None

    names = sorted(n for n in os.listdir(args.wav_dir)
                   if n.startswith("seg_") and n.endswith(".wav")
                   and int(n[4:9]) >= args.start_index)
    rej_dir = os.path.join(os.path.dirname(os.path.abspath(args.wav_dir)), "rejected")
    rejected = set()
    for n in names:
        path = os.path.join(args.wav_dir, n)
        ok, reason = clip_ok(path)
        if not ok:
            os.makedirs(rej_dir, exist_ok=True)
            shutil.move(path, os.path.join(rej_dir, n))
            rejected.add(os.path.abspath(path))
            eprint(f"  reject {n}: {reason}")

    if rejected:
        with open(args.meta, encoding="utf-8") as f:
            lines = f.read().splitlines()
        keep = [lines[0]] + [ln for ln in lines[1:]
                             if ln and os.path.abspath(ln.split("|", 1)[0]) not in rejected]
        with open(args.meta, "w", encoding="utf-8") as f:
            f.write("\n".join(keep) + "\n")

    eprint(f"Purity gate: kept {len(names) - len(rejected)}/{len(names)} clips "
           f"({len(rejected)} rejected -> {rej_dir if rejected else 'none'}).")


if __name__ == "__main__":
    main()
