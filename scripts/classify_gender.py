"""Classify which diarized speaker is female vs male by median pitch (F0).

Female voices sit at a higher median fundamental frequency than male. For a
2-speaker recording the higher-F0 speaker is female, the lower is male. Prints
"<female_idx> <male_idx>" (e.g. "0 1") to stdout; per-speaker F0 to stderr.

Runs in the F5 venv (librosa + soundfile). CPU only.
"""
import argparse
import json
import sys

import numpy as np
import soundfile as sf
import librosa


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def load_spans(path):
    with open(path, encoding="utf-8") as f:
        segs = json.load(f)["segments"]
    spans = {}
    for s in segs:
        if "start" in s and s.get("speaker"):
            spans.setdefault(s["speaker"], []).append((s["start"], s["end"]))
    return spans


def median_f0(audio, sr, spans, max_total=30.0):
    """Median voiced F0 over up to max_total seconds of the speaker's longest spans."""
    chunks, total = [], 0.0
    for a, b in sorted(spans, key=lambda x: -(x[1] - x[0])):
        if total >= max_total:
            break
        seg = audio[int(a * sr):int(b * sr)]
        if seg.size:
            chunks.append(seg)
            total += b - a
    if not chunks:
        return None
    y = np.concatenate(chunks).astype(np.float32)
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
    f0 = f0[~np.isnan(f0)]
    return float(np.median(f0)) if f0.size >= 10 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--female-min", type=float, default=165.0,
                    help="F0 threshold used only when a file has a single speaker")
    ap.add_argument("--min-gap", type=float, default=20.0,
                    help="min F0 gap (Hz) between the two speakers to trust the split; "
                         "below this the recording is skipped (prints '-1 -1')")
    args = ap.parse_args()

    audio, sr = sf.read(args.audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    spans = load_spans(args.json)
    f0s = {lab: median_f0(audio, sr, sp) for lab, sp in spans.items()}
    f0s = {k: v for k, v in f0s.items() if v is not None}

    eprint("Median F0 per speaker:")
    for lab in sorted(f0s):
        eprint(f"  {lab}: {f0s[lab]:.0f} Hz")
    if not f0s:
        eprint("ERROR: no voiced pitch found.")
        sys.exit(2)

    ordered = sorted(f0s.items(), key=lambda x: -x[1])   # highest F0 first
    if len(ordered) >= 2:
        gap = ordered[0][1] - ordered[-1][1]
        if gap < args.min_gap:
            # Too close to tell female from male reliably -- skip the whole
            # recording rather than risk assigning a male track to the female
            # dataset (or vice versa).
            eprint(f"SKIP: F0 gap {gap:.0f} Hz < {args.min_gap:.0f} Hz; gender split "
                   "unreliable -- excluding this recording from training.")
            female = male = None
        else:
            female, male = ordered[0][0], ordered[-1][0]
    else:
        lab, hz = ordered[0]
        if hz >= args.female_min:
            female, male = lab, None
        else:
            female, male = None, lab
        eprint(f"Only one speaker; classified as {'female' if female else 'male'} by threshold.")

    fem_idx = int(female.split("_")[1]) if female else -1
    male_idx = int(male.split("_")[1]) if male else -1
    eprint(f"-> female=SPEAKER_{fem_idx:02d} male=SPEAKER_{male_idx:02d}"
           .replace("SPEAKER_-1", "(none)"))
    print(f"{fem_idx} {male_idx}")   # stdout: "<female_idx> <male_idx>"


if __name__ == "__main__":
    main()
