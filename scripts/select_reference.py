"""Pick a clean single-speaker reference clip from a WhisperX diarized JSON.

Runs in the WhisperX venv (stdlib only + ffmpeg on PATH). Given the diarization
JSON and the source WAV, it builds windows from contiguous runs of the target
speaker's WORDS (broken whenever the other speaker interjects, so a window never
contains both voices), selects the best window, cuts it to a 24 kHz mono WAV,
and writes the matching reference text to <out>.txt (UTF-8).

Also prints a per-speaker talk-time summary to stderr so you can decide which
speaker to clone.
"""
import argparse
import json
import subprocess
import sys
from collections import Counter


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def load_segments(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("segments", [])


def speaker_of(seg):
    return seg.get("speaker")


def build_windows(segs, target, max_dur, pause):
    """Build <= max_dur windows from contiguous runs of the target's words.

    Uses WORD-level speaker labels, not segment labels: a single WhisperX segment
    is a long run-on that can straddle a turn exchange, so a segment-level window
    can contain the other voice. Here a window is broken whenever the other
    speaker talks in the gap to the next target word (their audio would otherwise
    land inside the continuous [start,end] slice), at a pause >= `pause`, or when
    the next word would overflow max_dur.
    """
    tgt, other = [], []
    for s in segs:
        for w in s.get("words", []):
            if not ("start" in w and "end" in w and w.get("word", "").strip()):
                continue
            sp = w.get("speaker")
            if sp == target:
                tgt.append(w)
            elif sp:
                other.append((w["start"], w["end"]))
    other.sort()

    def other_in_gap(g0, g1):
        if g1 <= g0:
            return False
        for a, b in other:
            if a >= g1:
                break
            if b > g0:
                return True
        return False

    def close(run, truncated):
        return {"start": run[0]["start"], "end": run[-1]["end"],
                "dur": run[-1]["end"] - run[0]["start"],
                "text": "".join(x["word"].strip() for x in run),
                "truncated": truncated}

    windows = []
    cur = []
    for i, w in enumerate(tgt):
        if cur and (w["end"] - cur[0]["start"] > max_dur):
            # length-forced break: this window likely ends MID-PHRASE. Mark it so
            # selection avoids it -- F5 mirrors the reference's ending, and a
            # ref_text cut mid-word skews the audio/text alignment.
            windows.append(close(cur, True)); cur = []
        cur.append(w)
        last = i + 1 == len(tgt)
        if last:
            windows.append(close(cur, False)); cur = []
        else:
            nxt = tgt[i + 1]["start"]
            if (nxt - w["end"]) >= pause or other_in_gap(w["end"], nxt):
                windows.append(close(cur, False)); cur = []
    return windows


def cps(w):
    """Speaking rate: non-space chars per second of the window's audio."""
    return len("".join(w["text"].split())) / max(w["dur"], 0.1)


def has_aabb(text):
    """True if the text contains an AABB reduplication (e.g. onomatopoeia like
    屁屁砰砰). Such spans are usually performed as sound effects / laughter, and
    F5-TTS mirrors that delivery into every synthesis, so demote these windows."""
    t = "".join(text.split())
    return any(a == b and c == d and a != c
               for a, b, c, d in zip(t, t[1:], t[2:], t[3:]))


def text_quality_ok(text, dur, max_cps):
    """Reject WhisperX hallucinations: abnormal char density or heavy repetition.

    A drilling/music/noise section transcribes as junk like a long run of the same
    character. Such text wrecks F5-TTS (it sizes output from the ref-text/ref-audio
    ratio), so we drop those windows.
    """
    t = "".join(text.split())
    if not t:
        return False
    if len(t) / max(dur, 0.1) > max_cps:          # too many chars per second
        return False
    longest_run = run = 1                          # longest run of one repeated char
    for a, b in zip(t, t[1:]):
        run = run + 1 if a == b else 1
        longest_run = max(longest_run, run)
    if longest_run > 6:
        return False
    if Counter(t).most_common(1)[0][1] / len(t) > 0.35:   # one char dominates
        return False
    return True


def summarize(segs):
    totals = {}
    for s in segs:
        spk = speaker_of(s)
        if spk and "start" in s:
            totals[spk] = totals.get(spk, 0.0) + (s["end"] - s["start"])
    eprint("Speaker talk-time:")
    for spk in sorted(totals):
        eprint(f"  {spk}: {totals[spk]:.1f}s")


def purity_pick(candidates, segs, target, args, max_try=None):
    """Return the first candidate window whose audio passes the acoustic purity
    gate. Word labels can miss overlapped speech, so the top-ranked window may
    still carry the other voice -- this verifies acoustically with two detectors:

    1. Identity margin (wespeaker) on 2s windows slid at 0.5s hops, STRICTER than
       filter_dataset.py's clip gate (other-sim >= 0.35 or lead < 0.30 rejects):
       a reference is a single clip F5 mirrors into every synthesis, so a brief
       bleed diluted inside one 2s window must still fail.
    2. Overlapped speech (pyannote segmentation-3.0, run once over the whole
       episode): reject any candidate with > --ref-max-overlap seconds of >=2
       simultaneously active speakers inside it. Catches simultaneous bleed the
       identity margin can't see.

    Returns None (caller falls back to rank order) if pyannote is unavailable
    or nothing passes."""
    try:
        import numpy as np
        import torch
        from pyannote.audio import Inference, Model
        from pyannote.core import Segment
        import os as _os
        tok = _os.environ.get("HF_TOKEN")

        def load(name):
            try:
                return Model.from_pretrained(name, use_auth_token=tok)
            except TypeError:
                return Model.from_pretrained(name, token=tok)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inf = Inference(load("pyannote/wespeaker-voxceleb-resnet34-LM"),
                        window="whole", device=device)

        def emb(a, b):
            v = np.asarray(inf.crop(args.audio, Segment(a, b))).reshape(-1)
            return v / (np.linalg.norm(v) + 1e-9)

        cent = {}
        for lab in sorted({s.get("speaker") for s in segs if s.get("speaker")}):
            sp = [(s["start"], s["end"]) for s in segs
                  if s.get("speaker") == lab and "start" in s and 3.0 <= s["end"] - s["start"] <= 15.0]
            sp.sort(key=lambda x: -(x[1] - x[0]))
            vs = [emb(a, b) for a, b in sp[:8]]
            if vs:
                c = np.mean(vs, axis=0)
                cent[lab] = c / (np.linalg.norm(c) + 1e-9)
        if target not in cent or len(cent) < 2:
            return None
        others = [c for lab, c in cent.items() if lab != target]

        # Overlapped-speech activity over the whole episode, computed once.
        # (pyannote 4.x removed pipelines.OverlappedSpeechDetection; decode the
        # multilabel output manually, max-aggregating the sliding chunks.)
        seg_inf = Inference(load("pyannote/segmentation-3.0"), device=device)
        out = seg_inf(args.audio)
        data = out.data if out.data.ndim == 3 else out.data[None]
        n_frames = data.shape[1]
        frame_dur = out.sliding_window.duration / n_frames
        step = out.sliding_window.step
        total = int(round(data.shape[0] * step / frame_dur)) + n_frames
        act = np.zeros((total, data.shape[2]))
        for c in range(data.shape[0]):
            off = int(round(c * step / frame_dur))
            end = min(off + n_frames, total)
            act[off:end] = np.maximum(act[off:end], data[c][:end - off])
        overlap_frames = (act > 0.5).sum(axis=1) >= 2

        def overlap_sec(a, b):
            i, j = int(a / frame_dur), int(b / frame_dur)
            return float(overlap_frames[i:j].sum() * frame_dur)

        def window_pure(a, b):
            ov = overlap_sec(a, b)
            if ov > args.ref_max_overlap:
                return False, f"overlapped speech {ov:.2f}s"
            t = a
            while True:
                e0, e1 = t, min(t + 2.0, b)
                if e1 - e0 < 1.0:
                    break
                e = emb(e0, e1)
                st = float(np.dot(e, cent[target]))
                so = max(float(np.dot(e, c)) for c in others)
                if so >= 0.35 or st - so < 0.30:
                    return False, f"other voice @{e0:.1f}s (target {st:.2f}, other {so:.2f})"
                if e1 >= b:
                    break
                t += 0.5
            return True, None

        pool = candidates if max_try is None else candidates[:max_try]
        for w in pool:
            ok, why = window_pure(w["start"], w["end"])
            if ok:
                return w
            eprint(f"  purity: skipping {w['start']:.1f}-{w['end']:.1f}s ({why})")
        eprint(f"WARN: none of the {len(pool)} candidate windows passed the "
               "purity check; falling back to rank order.")
        return None
    except Exception as ex:  # noqa: BLE001 -- purity is best-effort, selection must not die
        eprint(f"WARN: purity check unavailable ({ex}); using rank order.")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--speaker", required=True, help="0 or 1 (-> SPEAKER_00 / _01)")
    ap.add_argument("--out", required=True, help="output reference wav path")
    ap.add_argument("--min-dur", type=float, default=5.0)
    ap.add_argument("--max-dur", type=float, default=12.0)
    ap.add_argument("--pause", type=float, default=0.6,
                    help="inter-word gap (s) that ends a window (keeps refs to one utterance)")
    ap.add_argument("--ref-max-overlap", type=float, default=0.10,
                    help="max seconds of detected overlapped speech allowed in a "
                         "reference window (stricter than the dataset gate's 0.30)")
    ap.add_argument("--max-cps", type=float, default=9.0,
                    help="max chars/sec; above this a window is treated as noise/hallucination")
    args = ap.parse_args()

    segs = load_segments(args.json)
    summarize(segs)

    target = f"SPEAKER_{int(args.speaker):02d}"
    # Leave headroom under F5's hard limit: the cut adds +0.1s tail and 0.4s
    # silence padding, and a final wav OVER max_dur makes F5 clip the audio while
    # ref_text keeps the clipped words -- the model then SPEAKS those missing
    # words before the requested text (garbage prefix in every synthesis).
    cap = args.max_dur - 0.5
    windows = build_windows(segs, target, cap, args.pause)
    if not windows:
        eprint(f"ERROR: no segments found for {target}. Check --speaker / diarization.")
        sys.exit(2)

    in_range = [w for w in windows if args.min_dur <= w["dur"] <= cap]
    clean = [w for w in in_range if text_quality_ok(w["text"], w["dur"], args.max_cps)]
    if clean:
        # Prefer windows that end at a natural pause (not length-truncated): F5
        # mirrors the reference's ending, and mid-phrase cuts skew alignment.
        # Then prefer the window whose speaking rate is closest to the speaker's
        # median: F5 paces output from the ref's chars/sec, so a slow, pause-heavy
        # window makes every synthesis slow. Tie-break on longer duration.
        med_cps = sorted(cps(w) for w in clean)[len(clean) // 2]
        eprint(f"Median speaking rate over {len(clean)} clean windows: {med_cps:.2f} chars/s")
        clean.sort(key=lambda w: (w["truncated"], has_aabb(w["text"]),
                                  abs(cps(w) - med_cps), -w["dur"]))
        best = purity_pick(clean, segs, target, args) or clean[0]
    elif in_range:
        # nothing passed the noise filter; take the lowest text-density window
        best = min(in_range, key=lambda w: len("".join(w["text"].split())) / w["dur"])
        eprint("WARN: no clean window passed the noise filter; using lowest-density one.")
    else:
        best = max(windows, key=lambda w: w["dur"])
        eprint(f"WARN: no window in [{args.min_dur},{args.max_dur}]s; "
               f"using longest ({best['dur']:.1f}s).")

    eprint(f"Chosen {target} clip: {best['start']:.2f}-{best['end']:.2f}s "
           f"({best['dur']:.1f}s, {cps(best):.2f} chars/s): {best['text']}")

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", args.audio,
         # slight tail past the diarization end (avoids clipping the last word),
         # then append digital silence: F5-TTS copies the reference's ending style,
         # so a reference that ends in a clean pause yields non-abrupt output ends.
         "-ss", str(best["start"]), "-to", str(best["end"] + 0.1),
         "-af", "apad=pad_dur=0.4",
         "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", args.out],
        check=True,
    )
    with open(args.out + ".txt", "w", encoding="utf-8") as f:
        f.write(best["text"])
    eprint(f"Wrote {args.out} and {args.out}.txt")


if __name__ == "__main__":
    main()
