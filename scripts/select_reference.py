"""Pick a clean single-speaker reference clip from a WhisperX diarized JSON.

Runs in the WhisperX venv (stdlib only + ffmpeg on PATH). Given the diarization
JSON and the source WAV, it merges consecutive same-speaker segments into
windows, selects the best window for the target speaker, cuts it to a 24 kHz
mono WAV, and writes the matching reference text to <out>.txt (UTF-8).

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


def build_windows(segs, target, min_dur, max_dur):
    """Greedily merge consecutive target-speaker segments into <= max_dur windows."""
    windows = []
    i, n = 0, len(segs)
    while i < n:
        if speaker_of(segs[i]) != target or "start" not in segs[i]:
            i += 1
            continue
        start = segs[i]["start"]
        end = segs[i]["end"]
        parts = [segs[i].get("text", "").strip()]
        j = i + 1
        while j < n and speaker_of(segs[j]) == target:
            if segs[j]["end"] - start > max_dur:
                break
            end = segs[j]["end"]
            parts.append(segs[j].get("text", "").strip())
            j += 1
        text = "".join(parts).strip()
        windows.append({"start": start, "end": end, "dur": end - start, "text": text})
        i = max(j, i + 1)
    return windows


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--speaker", required=True, help="0 or 1 (-> SPEAKER_00 / _01)")
    ap.add_argument("--out", required=True, help="output reference wav path")
    ap.add_argument("--min-dur", type=float, default=5.0)
    ap.add_argument("--max-dur", type=float, default=12.0)
    ap.add_argument("--max-cps", type=float, default=9.0,
                    help="max chars/sec; above this a window is treated as noise/hallucination")
    args = ap.parse_args()

    segs = load_segments(args.json)
    summarize(segs)

    target = f"SPEAKER_{int(args.speaker):02d}"
    windows = build_windows(segs, target, args.min_dur, args.max_dur)
    if not windows:
        eprint(f"ERROR: no segments found for {target}. Check --speaker / diarization.")
        sys.exit(2)

    in_range = [w for w in windows if args.min_dur <= w["dur"] <= args.max_dur]
    clean = [w for w in in_range if text_quality_ok(w["text"], w["dur"], args.max_cps)]
    if clean:
        # F5-TTS clips reference to ~12s, so among clean windows prefer the
        # longest (up to max_dur); tie-break on more text.
        best = sorted(clean, key=lambda w: (-w["dur"], -len(w["text"])))[0]
    elif in_range:
        # nothing passed the noise filter; take the lowest text-density window
        best = min(in_range, key=lambda w: len("".join(w["text"].split())) / w["dur"])
        eprint("WARN: no clean window passed the noise filter; using lowest-density one.")
    else:
        best = max(windows, key=lambda w: w["dur"])
        eprint(f"WARN: no window in [{args.min_dur},{args.max_dur}]s; "
               f"using longest ({best['dur']:.1f}s).")

    eprint(f"Chosen {target} clip: {best['start']:.2f}-{best['end']:.2f}s "
           f"({best['dur']:.1f}s): {best['text']}")

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", args.audio,
         "-ss", str(best["start"]), "-to", str(best["end"]),
         "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", args.out],
        check=True,
    )
    with open(args.out + ".txt", "w", encoding="utf-8") as f:
        f.write(best["text"])
    eprint(f"Wrote {args.out} and {args.out}.txt")


if __name__ == "__main__":
    main()
