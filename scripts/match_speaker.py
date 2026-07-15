"""Auto-match a target speaker across recordings via voice embeddings.

Diarization labels (SPEAKER_00/01) are assigned independently per file, so the
"same" person has different labels in different recordings. Given a reference
(recording + known target label), this finds which label in another recording is
the same voice, using the cached pyannote wespeaker embedding model.

Runs in the WhisperX venv (has pyannote.audio). Forced to CPU so it never
contends with a training GPU. Prints the matched label index (0/1) to stdout;
similarity details go to stderr.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def load_segs(path, speaker=None):
    with open(path, encoding="utf-8") as f:
        segs = json.load(f)["segments"]
    return [
        (s["speaker"], s["start"], s["end"])
        for s in segs
        if "start" in s and s.get("speaker") and (speaker is None or s["speaker"] == speaker)
    ]


def labels_in(path):
    with open(path, encoding="utf-8") as f:
        segs = json.load(f)["segments"]
    return sorted({s["speaker"] for s in segs if s.get("speaker")})


def top_segments(segs, k=6, min_dur=3.0, max_dur=15.0):
    cand = [(spk, a, b) for spk, a, b in segs if min_dur <= b - a <= max_dur]
    cand.sort(key=lambda x: -(x[2] - x[1]))
    return cand[:k]


def embed(inference, audio, segs):
    from pyannote.core import Segment

    vecs = []
    for _spk, a, b in top_segments(segs):
        try:
            e = np.asarray(inference.crop(audio, Segment(a, b))).reshape(-1)
            vecs.append(e)
        except Exception as ex:  # noqa: BLE001
            eprint(f"  skip seg {a:.1f}-{b:.1f}: {ex}")
    if not vecs:
        return None
    v = np.mean(vecs, axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--ref-json", required=True)
    ap.add_argument("--ref-speaker", required=True, help="0 or 1: the known target in the reference")
    ap.add_argument("--audio", required=True, help="target recording audio")
    ap.add_argument("--json", required=True, help="target recording diarization")
    args = ap.parse_args()

    from pyannote.audio import Inference, Model

    tok = os.environ.get("HF_TOKEN")
    try:
        model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", use_auth_token=tok)
    except TypeError:
        model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", token=tok)
    inference = Inference(model, window="whole", device=torch.device("cpu"))

    ref_label = f"SPEAKER_{int(args.ref_speaker):02d}"
    ref_emb = embed(inference, args.ref_audio, load_segs(args.ref_json, ref_label))
    if ref_emb is None:
        eprint(f"ERROR: no usable segments for reference {ref_label}.")
        sys.exit(2)

    scored = []
    for lab in labels_in(args.json):
        emb = embed(inference, args.audio, load_segs(args.json, lab))
        if emb is None:
            continue
        scored.append((lab, float(np.dot(ref_emb, emb))))
    if not scored:
        eprint("ERROR: no usable speakers in target.")
        sys.exit(2)

    scored.sort(key=lambda x: -x[1])
    best_label, best_sim = scored[0]
    margin = best_sim - (scored[1][1] if len(scored) > 1 else -1.0)
    eprint("Similarity to reference:")
    for lab, sim in scored:
        eprint(f"  {lab}: {sim:.3f}")
    eprint(f"-> matched {best_label} (sim {best_sim:.3f}, margin {margin:.3f})")
    if best_sim < 0.5:
        eprint("WARN: low similarity; the target speaker may not be present in this recording.")

    print(int(best_label.split("_")[1]))   # stdout: label index for the orchestrator


if __name__ == "__main__":
    main()
