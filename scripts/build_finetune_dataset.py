"""Build an F5-TTS fine-tuning dataset from a WhisperX diarized JSON.

Extracts every clean single-speaker segment for the target speaker as an
individual 24 kHz mono clip and writes a metadata.csv (`wav_path|text`) that
`prepare_csv_wavs.py` consumes. Reuses the same noise/hallucination filter as
select_reference.py so drilling/music/garbled segments are dropped.

Runs in the F5 venv (needs soundfile + numpy).
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import soundfile as sf


def eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def text_quality_ok(text, dur, max_cps):
    t = "".join(text.split())
    if not t:
        return False
    if len(t) / max(dur, 0.1) > max_cps:
        return False
    longest = run = 1
    for a, b in zip(t, t[1:]):
        run = run + 1 if a == b else 1
        longest = max(longest, run)
    if longest > 6:
        return False
    if Counter(t).most_common(1)[0][1] / len(t) > 0.35:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--speaker", required=True, help="0 or 1 (-> SPEAKER_00 / _01)")
    ap.add_argument("--out-dir", required=True, help="dataset staging dir (gets wavs/ + metadata.csv)")
    ap.add_argument("--target-min", type=float, default=3.0, help="min clip length before a pause can break it")
    ap.add_argument("--target-max", type=float, default=12.0, help="force a break at this clip length")
    ap.add_argument("--pause", type=float, default=0.3, help="inter-word gap (s) that counts as a natural break")
    ap.add_argument("--min-dur", type=float, default=2.0, help="drop clips shorter than this")
    ap.add_argument("--max-dur", type=float, default=15.0)
    ap.add_argument("--max-cps", type=float, default=9.0)
    ap.add_argument("--append", action="store_true",
                    help="add to an existing dataset dir (continue numbering, append metadata) "
                         "-- for combining multiple recordings of the same speaker")
    args = ap.parse_args()

    target = f"SPEAKER_{int(args.speaker):02d}"
    with open(args.json, encoding="utf-8") as f:
        segs = json.load(f).get("segments", [])

    # Flatten to the target speaker's words (word-level timestamps), in order,
    # and collect the OTHER speakers' word intervals so we can keep their audio
    # out of the target's clips.
    words = []
    other_iv = []
    for s in segs:
        for w in s.get("words", []):
            if not ("start" in w and "end" in w and w.get("word", "").strip()):
                continue
            sp = w.get("speaker")
            if sp == target:
                words.append(w)
            elif sp:
                other_iv.append((w["start"], w["end"]))
    other_iv.sort()

    def other_in_gap(g0, g1):
        # True if any other-speaker word overlaps the open gap (g0, g1) between
        # two consecutive target words. A continuous audio slice across such a
        # gap would bake the other speaker into the clip.
        if g1 <= g0:
            return False
        for a, b in other_iv:
            if a >= g1:
                break
            if b > g0:
                return True
        return False

    # Chunk words into clips: break at a natural pause once past target_min, or
    # force a break at target_max. CRITICALLY, always break when the other
    # speaker talks in the gap to the next target word -- otherwise the [start,
    # end] slice would include their voice (measured 9-26% of audio otherwise).
    # Breaking there drops that turn into the inter-clip gap, never a clip.
    clips = []            # (start, end, text)
    cur = []
    for i, w in enumerate(words):
        cur.append(w)
        dur = w["end"] - cur[0]["start"]
        if i + 1 < len(words):
            nxt = words[i + 1]["start"]
            gap_next = nxt - w["end"]
            interrupted = other_in_gap(w["end"], nxt)
        else:
            gap_next = 1e9
            interrupted = True
        if dur >= args.target_max or interrupted or (dur >= args.target_min and gap_next >= args.pause):
            clips.append((cur[0]["start"], w["end"], "".join(x["word"].strip() for x in cur)))
            cur = []
    if cur:
        clips.append((cur[0]["start"], cur[-1]["end"], "".join(x["word"].strip() for x in cur)))

    audio, sr = sf.read(args.audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    import glob as _glob
    wav_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    # In append mode, continue numbering past the highest existing index so nothing
    # is overwritten (a count would collide after the purity gate removes clips).
    start_idx = 0
    if args.append:
        existing = _glob.glob(os.path.join(wav_dir, "seg_*.wav"))
        if existing:
            start_idx = max(int(os.path.basename(p)[4:9]) for p in existing) + 1
    rows = []
    total = 0.0
    kept = skipped = 0
    for start, end, text in clips:
        dur = end - start
        if not (args.min_dur <= dur <= args.max_dur) or not text_quality_ok(text, dur, args.max_cps):
            skipped += 1
            continue
        clip = audio[int(start * sr):int(end * sr)]
        if clip.size == 0:
            skipped += 1
            continue
        name = f"seg_{start_idx + kept:05d}.wav"
        path = os.path.join(wav_dir, name)
        sf.write(path, clip, sr, subtype="PCM_16")
        rows.append(f"{os.path.abspath(path)}|{text}")
        total += dur
        kept += 1

    if kept == 0:
        eprint(f"ERROR: no clean {target} segments found.")
        sys.exit(2)

    meta = os.path.join(args.out_dir, "metadata.csv")
    exists = os.path.isfile(meta)
    with open(meta, "a" if (args.append and exists) else "w", encoding="utf-8") as f:
        if not (args.append and exists):
            f.write("audio_file|text\n")       # header required by prepare_csv_wavs
        f.write("\n".join(rows) + "\n")

    eprint(f"{target}: kept {kept} clips ({total/60:.1f} min), skipped {skipped}"
           f"{' [appended]' if args.append else ''}.")
    eprint(f"Wrote {meta}")


if __name__ == "__main__":
    main()
