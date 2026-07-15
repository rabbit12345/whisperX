"""Analyze a YouTube playlist and classify each clip as usable for training.

The show has two regular hosts. Clips that also feature a guest voice must NOT
be used for training. By convention such titles carry a guest marker, most often
"ft." / "feat." followed by the guest name(s). This module classifies each clip:

  status = "include"    -> two hosts only, safe to use for training
           "exclude"    -> a guest marker was found (ft./feat./featuring/w/ ...)
           "uncertain"  -> a soft/ambiguous marker (with/guest/...) -> review by hand

The classifier is pure text analysis (no network), so it is deterministic and
unit-testable. Fetching titles is a separate, pluggable step:

  * --playlist-json FILE : read a JSON array of clips (each needs id + title).
                           This is what `yt-dlp --flat-playlist -J URL` produces
                           (entries[]), or what the existing downloader project
                           can hand us. No network, no extra dependency.
  * --url URL            : fetch the playlist with urllib (stdlib only, no yt-dlp);
                           scrapes ytInitialData + innertube continuation paging.

Output: a JSON object { playlist, analyzed_at, counts, clips:[...] } written to
--out (default stdout). Each clip keeps its id/title/duration/upload_date plus
the classification (status, markers, guests) and a "selected" flag the frontend
toggles (defaults to true for "include", false otherwise).
"""
import argparse
import datetime as _dt
import json
import re
import sys
import urllib.request

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


# Strong markers: their presence means a guest is almost certainly featured.
# Word-boundary anchored; ft/feat may be followed by an optional dot.
STRONG = re.compile(r"(?<![a-z])(ft|feat|featuring)\.?(?![a-z])|(?<!\w)w/\s", re.I)
# Soft markers: ambiguous. Could be a guest, could be topical ("with the boys").
SOFT = re.compile(r"(?<!\w)(with|guest|joined by|special guest| w/)(?!\w)", re.I)
# Where a guest name tends to start, for extraction/display.
GUEST_SPLIT = re.compile(r"(?<![a-z])(?:ft|feat|featuring)\.?(?![a-z])|(?<!\w)w/\s"
                         r"|(?<!\w)(?:with|guest|joined by)(?!\w)", re.I)


def extract_guests(title):
    """Best-effort guest name(s): the text after the first guest marker, trimmed
    at a trailing separator (|, -, (, )). Only for display/review, not decisions."""
    m = GUEST_SPLIT.search(title)
    if not m:
        return ""
    tail = title[m.end():]
    tail = re.split(r"[|\-–—()\[\]]", tail, maxsplit=1)[0]
    return tail.strip(" :,-").strip()


def classify(title):
    """Return (status, markers, guests) for one title."""
    t = title or ""
    strong = sorted({m.group(0).strip().lower() for m in STRONG.finditer(t)})
    soft = sorted({m.group(0).strip().lower() for m in SOFT.finditer(t)})
    if strong:
        return "exclude", strong, extract_guests(t)
    if soft:
        return "uncertain", soft, extract_guests(t)
    return "include", [], ""


# --- playlist fetch (stdlib only; no yt-dlp) ---------------------------------
# YouTube's playlist page embeds ytInitialData with one lockupViewModel per video
# (contentId = videoId, metadata.lockupMetadataViewModel.title.content = title).
# Playlists over 100 items page via the innertube browse continuation endpoint.

# Consent cookie: without it, EU-routed requests return empty continuations.
_COOKIE = "SOCS=CAI; CONSENT=YES+1"


def _http_get(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9", "Cookie": _COOKIE})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")


def _http_post_json(url, payload, visitor=None):
    body = json.dumps(payload).encode("utf-8")
    headers = {"User-Agent": _UA, "Content-Type": "application/json", "Cookie": _COOKIE}
    if visitor:
        headers["X-Goog-Visitor-Id"] = visitor
    req = urllib.request.Request(url, data=body, headers=headers)
    return json.loads(urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace"))


def _dur_to_sec(t):
    if not t:
        return None
    s = 0
    for p in t.split(":"):
        s = s * 60 + int(p)
    return s


def _badge_len(lvm):
    for ov in (lvm.get("contentImage", {}).get("thumbnailViewModel", {}).get("overlays") or []):
        for b in (ov.get("thumbnailBottomOverlayViewModel", {}).get("badges") or []):
            t = b.get("thumbnailBadgeViewModel", {}).get("text", "")
            if re.match(r"^\d+(:\d+)+$", t):
                return t
    return None


def _find_token(node):
    """Deep-search a node for a continuation token."""
    if isinstance(node, dict):
        cc = node.get("continuationCommand")
        if isinstance(cc, dict) and cc.get("token"):
            return cc["token"]
        for v in node.values():
            r = _find_token(v)
            if r:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _find_token(v)
            if r:
                return r
    return None


def _collect(node, out):
    """Append every video lockup under node to out; return the NEXT-PAGE continuation
    token. The correct token is the continuationItem that sits in the SAME list as the
    video lockups (a bare _find_token would grab an unrelated reload/section token that
    just re-returns page 1)."""
    token = None
    if isinstance(node, dict):
        lvm = node.get("lockupViewModel")
        if isinstance(lvm, dict) and lvm.get("contentId"):
            title = (lvm.get("metadata", {}).get("lockupMetadataViewModel", {})
                     .get("title", {}).get("content", ""))
            out.append({"video_id": lvm["contentId"], "title": title,
                        "duration_sec": _dur_to_sec(_badge_len(lvm))})
        for v in node.values():
            token = _collect(v, out) or token
    elif isinstance(node, list):
        has_lockups = any(isinstance(e, dict) and "lockupViewModel" in e for e in node)
        for e in node:
            token = _collect(e, out) or token
            if has_lockups and isinstance(e, dict) and \
                    ("continuationItemRenderer" in e or "continuationItemViewModel" in e):
                token = _find_token(e) or token
    return token


def fetch_playlist(url):
    """Return (clips, playlist_title) for a YouTube playlist URL using only urllib."""
    html = _http_get(url)
    m = (re.search(r"var ytInitialData\s*=\s*(\{.*?\});</script>", html)
         or re.search(r'ytInitialData"?\]?\s*=\s*(\{.*?\});</script>', html))
    if not m:
        sys.exit("Could not find ytInitialData on the playlist page (layout changed or blocked).")
    data = json.loads(m.group(1))
    api_key = (re.search(r'"INNERTUBE_API_KEY":"([^"]+)"', html) or [None, None])[1]
    cver = (re.search(r'"INNERTUBE_CLIENT_VERSION":"([^"]+)"', html) or [None, "2.20240101.00.00"])[1]
    vm = re.search(r'"visitorData":"([^"]+)"', html)
    visitor = vm.group(1) if vm else None
    tm = re.search(r"<title>([^<]+)</title>", html)
    pl_title = tm.group(1).replace(" - YouTube", "").strip() if tm else url

    client = {"clientName": "WEB", "clientVersion": cver}
    if visitor:
        client["visitorData"] = visitor

    out = []
    token = _collect(data, out)
    seen = {v["video_id"] for v in out}
    pages = 0
    while token and api_key and pages < 60:      # 60*100 = 6000 videos safety cap
        pages += 1
        resp = _http_post_json(
            f"https://www.youtube.com/youtubei/v1/browse?key={api_key}",
            {"context": {"client": client}, "continuation": token}, visitor=visitor)
        page = []
        token = _collect(resp, page)
        added = 0
        for v in page:
            if v["video_id"] not in seen:
                seen.add(v["video_id"])
                out.append(v)
                added += 1
        if added == 0:                            # no progress -> stop
            break
    return out, pl_title


def _get(entry, *keys, default=None):
    for k in keys:
        if entry.get(k) not in (None, ""):
            return entry[k]
    return default


def normalize(entry):
    """Map a raw clip dict (our schema OR a yt-dlp entry) to a flat clip record."""
    vid = _get(entry, "video_id", "id", "url")
    title = _get(entry, "title", default="")
    dur = _get(entry, "duration_sec", "duration")
    upload = _get(entry, "upload_date")
    status, markers, guests = classify(title)
    return {
        "video_id": vid,
        "title": title,
        "duration_sec": dur,
        "upload_date": upload,
        "status": status,
        "markers": markers,
        "guests": guests,
        "selected": status == "include",   # frontend default; user can toggle
    }


def load_clips(args):
    if args.playlist_json:
        with open(args.playlist_json, encoding="utf-8") as f:
            data = json.load(f)
        # accept either a bare list or a yt-dlp -J object with "entries"
        entries = data.get("entries", data) if isinstance(data, dict) else data
        title = data.get("title") if isinstance(data, dict) else None
        return entries, (title or args.playlist_json)
    if args.url:
        return fetch_playlist(args.url)          # stdlib scrape (no yt-dlp)
    sys.exit("Provide --playlist-json FILE or --url URL (or --self-test).")


def analyze(entries, playlist_name):
    clips = [normalize(e) for e in entries if e]
    counts = {"include": 0, "exclude": 0, "uncertain": 0}
    for c in clips:
        counts[c["status"]] += 1
    return {
        "playlist": playlist_name,
        "analyzed_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "counts": {"total": len(clips), **counts},
        "clips": clips,
    }


SELFTEST = [
    ("How to lose money fast", "include"),
    ("Q3 recap ft. Jane Doe", "exclude"),
    ("Big episode feat. John & Sarah", "exclude"),
    ("Deep dive (featuring a special guest)", "exclude"),
    ("Market chaos w/ Mark Cuban", "exclude"),
    ("Hanging with the boys", "uncertain"),
    ("Our guest rules for 2026", "uncertain"),
    ("Software vs hardware debate", "include"),
    ("The gift that keeps giving", "include"),   # 'gift' must not trip 'ft'
    ("Left behind: a retrospective", "include"),  # 'left' must not trip 'ft'
]


def self_test():
    ok = True
    for title, expected in SELFTEST:
        got, markers, guests = classify(title)
        flag = "OK " if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"[{flag}] {got:9} (exp {expected:9}) guests={guests!r:20} :: {title}")
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--playlist-json", help="JSON array of clips, or a yt-dlp -J object")
    ap.add_argument("--url", help="playlist URL (needs yt-dlp installed)")
    ap.add_argument("--out", help="write result JSON here (default: stdout)")
    ap.add_argument("--self-test", action="store_true", help="run the classifier test cases")
    args = ap.parse_args()

    # Titles are non-ASCII; make console output UTF-8 so summary prints never crash
    # on a cp1252 Windows console regardless of the caller's environment.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    if args.self_test:
        self_test()

    entries, name = load_clips(args)
    result = analyze(entries, name)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        c = result["counts"]
        print(f"{name}: {c['total']} clips -> include={c['include']} "
              f"exclude={c['exclude']} uncertain={c['uncertain']}  ({args.out})")
    else:
        print(text)


if __name__ == "__main__":
    main()
