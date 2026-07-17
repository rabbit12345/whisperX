"""Local dashboard + trigger backend for the voice-clone training pipeline.

Zero external dependencies (Python stdlib only). Serves a single-page UI and a
small JSON API that reads the training history / playlist analysis and can
trigger three pipeline actions as background jobs:

  * analyze  -> scripts/analyze_playlist.py  (classify a playlist)
  * download -> config["download_cmd"] per video id  (the existing downloader
                project; command template configured in config.json)
  * retrain  -> scripts/Invoke-VoiceClones.ps1 (fresh) or Invoke-Finetune.ps1
                with -AutoMatch (append new MKVs to the running s0/s1 datasets)

Run:  python webui/server.py [--port 8756]
Then open the printed URL. Everything runs locally against your own files.
"""
import argparse
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def cfg(key, default=""):
    return CONFIG.get(key) or default


def work_dir():
    return Path(cfg("work_dir"))


def history_path():
    return Path(cfg("history_file") or (work_dir() / "training_history.json"))


def analysis_path():
    return Path(cfg("analysis_file") or (work_dir() / "playlist_analysis.json"))


def jobs_dir():
    d = work_dir() / "webui_jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def downloads_index_path():
    return work_dir() / "downloads_index.json"


def _load_downloads_index():
    f = downloads_index_path()
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {}


def _update_downloads_index(new_entries):
    if not new_entries:
        return
    idx = _load_downloads_index()
    idx.update(new_entries)
    try:
        downloads_index_path().write_text(json.dumps(idx, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    except OSError:
        pass


# --- training queue: downloaded-but-untrained MKVs staged for the next retrain ---
def queue_path():
    return work_dir() / "training_queue.json"


def _load_queue():
    """Queue items [{mkv, video_id, title}], auto-pruned of anything already trained."""
    f = queue_path()
    items = []
    if f.exists():
        try:
            items = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            items = []
    trained = {_norm(p) for p in trained_mkvs()}
    for it in items:                       # heal stale paths (e.g. old _N collision suffix)
        if it.get("mkv"):
            it["mkv"] = _resolve_mkv(it["mkv"])
    return [it for it in items if it.get("mkv") and _norm(it["mkv"]) not in trained]


def _save_queue(items):
    seen, out = set(), []
    for it in items:
        p = it.get("mkv")
        if p and _norm(p) not in seen:
            seen.add(_norm(p))
            out.append(it)
    try:
        queue_path().write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    return out


# --- job runner --------------------------------------------------------------
JOBS = {}          # id -> {id, kind, cmd, status, started, ended, log}
JOBS_LOCK = threading.Lock()

# Training jobs mutate the shared checkpoint/dataset dirs, so they must never run
# concurrently (that spawns multiple trainer processes and, for reset-retrain,
# wipes state another run is using). These kinds go through a single FIFO worker
# instead of each spawning its own thread; everything else (download, analyze)
# still runs concurrently.
TRAINING_KINDS = {"retrain", "retrain-append", "reset-retrain"}
TRAIN_QUEUE = queue.Queue()
TRAIN_PAUSED = threading.Event()   # set() => worker stops starting the NEXT job


def _train_worker():
    """Run queued training jobs strictly one at a time, in submission order.
    While paused, no new job is started (a job already running is left alone);
    jobs cancelled while queued are skipped when reached."""
    while True:
        # Poll (not a blocking get) so a pause toggled while we'd otherwise be
        # blocked on the queue is always honored before the next job is pulled.
        if TRAIN_PAUSED.is_set():
            time.sleep(0.2)
            continue
        try:
            job_id, argv, shell_cmd, env = TRAIN_QUEUE.get_nowait()
        except queue.Empty:
            time.sleep(0.2)
            continue
        job = JOBS.get(job_id)
        if job is not None and job.get("status") == "queued":
            job["status"] = "running"
            job["started"] = datetime.now().isoformat(timespec="seconds")
            _persist_jobs()
            _run(job_id, argv, shell_cmd, env)
        TRAIN_QUEUE.task_done()


def _kill_tree(pid):
    """Kill a launched job's whole process tree (pwsh -> python trainer)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True)
        else:
            os.kill(pid, 15)
    except OSError:
        pass


def _cancel_job(job):
    """Stop a queued or running training job. Returns a short status note."""
    st = job.get("status")
    if st == "queued":
        job["status"] = "cancelled"
        job["ended"] = datetime.now().isoformat(timespec="seconds")
        _persist_jobs()
        return "removed from queue"
    if st == "running":
        job["cancelled"] = True            # _run reads this to record "cancelled"
        if job.get("pid"):
            _kill_tree(job["pid"])
        return "stopping running job"
    return f"job already {st}"


def _run(job_id, argv, shell_cmd=None, env=None):
    job = JOBS[job_id]
    logf = jobs_dir() / f"{job_id}.log"
    job["log"] = str(logf)
    with open(logf, "w", encoding="utf-8", errors="replace") as lf:
        lf.write(f"$ {shell_cmd or ' '.join(argv)}\n\n")
        lf.flush()
        try:
            p = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT,
                                 cwd=cfg("scripts_dir"), env=env, text=True)
            job["pid"] = p.pid
            rc = p.wait()
            job["status"] = "cancelled" if job.get("cancelled") else \
                ("done" if rc == 0 else "failed")
            job["returncode"] = rc
        except Exception as e:  # noqa: BLE001
            lf.write(f"\n[launcher error] {e}\n")
            job["status"] = "failed"
            job["returncode"] = -1
    # For download jobs, capture each `--json` result line. The downloader's result
    # dict carries status/video_id/title/final_path, so we record structured items
    # (shown in the processing history) plus the produced paths and the id->file index.
    # A "YTDL_ID=<id>" marker printed before each call is a fallback id source.
    if job["kind"] == "download":
        produced, items, index, cur = [], [], {}, None
        for line in logf.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("YTDL_ID="):
                cur = line[len("YTDL_ID="):]
            elif line.startswith("{"):
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                vid = d.get("video_id") or cur
                title = d.get("title") or (os.path.splitext(os.path.basename(d["final_path"]))[0]
                                           if d.get("final_path") else None)
                items.append({"video_id": vid, "title": title,
                              "final_path": d.get("final_path"), "status": d.get("status")})
                if d.get("status") == "done" and d.get("final_path"):
                    produced.append(d["final_path"])
                    if vid:
                        index[vid] = {"final_path": d["final_path"], "title": title,
                                      "at": datetime.now().isoformat(timespec="seconds")}
        job["produced"] = produced
        job["items"] = items
        _update_downloads_index(index)
    # After a from-scratch retrain, repoint the TTS reference to the newest clip.
    if job["kind"] == "reset-retrain" and job.get("status") == "done" and job.get("newest_mkv"):
        note = _update_tts_reference(job["newest_mkv"])
        with open(logf, "a", encoding="utf-8", errors="replace") as lf:
            lf.write(f"\n[reference] {note}\n")
    job["ended"] = datetime.now().isoformat(timespec="seconds")
    _persist_jobs()


def start_job(kind, argv, shell_cmd=None, env=None, meta=None):
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    serialized = kind in TRAINING_KINDS
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id, "kind": kind, "cmd": shell_cmd or " ".join(argv),
            # Training jobs wait in the FIFO until the worker picks them up.
            "status": "queued" if serialized else "running",
            "started": datetime.now().isoformat(timespec="seconds"),
            "ended": None, "returncode": None,
            **(meta or {}),
        }
    if serialized:
        TRAIN_QUEUE.put((job_id, argv, shell_cmd, env))
    else:
        threading.Thread(target=_run, args=(job_id, argv, shell_cmd, env), daemon=True).start()
    _persist_jobs()
    return JOBS[job_id]


def _persist_jobs():
    try:
        (jobs_dir() / "jobs.json").write_text(
            json.dumps(list(JOBS.values()), indent=2), encoding="utf-8")
    except OSError:
        pass


def _load_jobs():
    f = jobs_dir() / "jobs.json"
    if f.exists():
        try:
            for j in json.loads(f.read_text(encoding="utf-8")):
                # anything left "running" (or still "queued" in the in-memory
                # FIFO, which didn't survive the restart) is stale
                if j.get("status") in ("running", "queued"):
                    j["status"] = "unknown"
                JOBS[j["id"]] = j
        except (OSError, ValueError):
            pass


# --- pipeline command builders -----------------------------------------------
def ps_array(items):
    """Render a PowerShell string[] literal with each item single-quoted."""
    return "@(" + ",".join("'" + str(i).replace("'", "''") + "'" for i in items) + ")"


def _norm(p):
    return os.path.normcase(os.path.abspath(p))


def _resolve_mkv(path):
    """Map a recorded MKV path to a file that actually exists: return it if present;
    else retry without a trailing _N collision suffix (a leftover of the downloader's
    old duplicate naming); else return the original so the caller reports it missing."""
    if not path or os.path.exists(path):
        return path
    d, base = os.path.split(path)
    stem, ext = os.path.splitext(base)
    m = re.match(r"^(.*?)_\d+$", stem)
    if m:
        cand = os.path.join(d, m.group(1) + ext)
        if os.path.exists(cand):
            return cand
    return path


def read_history():
    """Training history as a list, tolerant of a lone-object file (PowerShell's
    ConvertTo-Json collapses a 1-element array into a bare object)."""
    h = read_json_file(history_path())
    if isinstance(h, dict):
        return [h]
    return h if isinstance(h, list) else []


def trained_mkvs():
    """Distinct source MKV paths across every recorded training run. Basis for the
    append-union and for duplicate detection (history records sources[].mkv)."""
    seen, out = set(), []
    for rec in read_history():
        if not isinstance(rec, dict):
            continue
        for s in rec.get("sources", []) or []:
            p = s.get("mkv")
            if p and _norm(p) not in seen:
                seen.add(_norm(p))
                out.append(p)
    return out


def _trained_video_ids():
    """Video ids whose downloaded file has already been used in a training run."""
    tset = {_norm(p) for p in trained_mkvs()}
    idx = _load_downloads_index()
    return [vid for vid, meta in idx.items()
            if meta.get("final_path") and _norm(meta["final_path"]) in tset]


# --- live training progress + GPU stats -------------------------------------
# tqdm line from the trainer, e.g.:
#   Epoch 100/100:  83%|... | 232/281 [01:00<00:08,  5.50update/s, loss=0.866, update=28050]
_TQDM_RE = re.compile(r"Epoch\s+(\d+)/(\d+):\s+\d+%\|.*?\|\s*(\d+)/(\d+)\s*\["
                      r"[\d:]+<[^,\]]+,\s*([\d.]+)update/s")
_LOSS_RE = re.compile(r"loss=([\d.]+)")
# stage marker printed by Invoke-VoiceClones.ps1: --- Female -> s0  (N recording(s)) ---
_STAGE_RE = re.compile(r"---\s+(Female|Male)\s+->\s+(\S+)")
# incremental log-scan state per training job: {"offset", "gender", "dataset", "stage_num", "line"}
_PROG_CACHE = {}


def training_progress():
    """Parse the running training job's log tail into structured progress:
    current stage (Female/Male + dataset), tqdm epoch/step/rate/loss, and a
    whole-job ETA (rest of this epoch + remaining epochs + remaining stage,
    assuming the other voice's epochs take about as long)."""
    job = next((j for j in JOBS.values()
                if j.get("kind") in TRAINING_KINDS and j.get("status") == "running"), None)
    if not job:
        _PROG_CACHE.clear()
        return None
    logf = job.get("log")
    if not logf or not Path(logf).exists():
        return {"job_id": job["id"], "kind": job["kind"]}
    st = _PROG_CACHE.setdefault(job["id"], {"offset": 0, "gender": None, "dataset": None,
                                            "stage_num": 0, "line": ""})
    try:
        with open(logf, "rb") as f:
            f.seek(st["offset"])
            new = f.read().decode("utf-8", "replace")
            st["offset"] = f.tell()
    except OSError:
        new = ""
    if new:
        # tqdm rewrites its line with \r; treat every rewrite as its own line
        lines = new.replace("\r", "\n").splitlines()
        for ln in lines:
            m = _STAGE_RE.search(ln)
            if m:
                st["gender"], st["dataset"] = m.group(1), m.group(2)
                st["stage_num"] += 1
        for ln in reversed(lines):
            if _TQDM_RE.search(ln):
                st["line"] = ln.strip()
                break
    cmd = job.get("cmd", "")
    total_stages = 1 if ("-Train female" in cmd or "-Train male" in cmd) else 2
    out = {"job_id": job["id"], "kind": job["kind"], "gender": st["gender"],
           "dataset": st["dataset"], "stage_num": st["stage_num"] or None,
           "total_stages": total_stages, "line": st["line"][-160:] or None}
    m = _TQDM_RE.search(st["line"])
    if m:
        epoch, epochs, step, steps = (int(m.group(i)) for i in range(1, 5))
        rate = float(m.group(5))
        out.update(epoch=epoch, epochs=epochs, step=step, steps=steps, rate=rate)
        lm = _LOSS_RE.search(st["line"])
        if lm:
            out["loss"] = float(lm.group(1))
        if rate > 0:
            remaining = (steps - step) + (epochs - epoch) * steps
            if st["stage_num"] and st["stage_num"] < total_stages:
                remaining += (total_stages - st["stage_num"]) * epochs * steps
            out["eta_seconds"] = int(remaining / rate)
    return out


_GPU_CACHE = {"t": 0.0, "v": None}


def gpu_stats():
    """Utilization % + VRAM from nvidia-smi, cached ~2s so polling stays cheap."""
    now = time.time()
    if now - _GPU_CACHE["t"] < 2:
        return _GPU_CACHE["v"]
    v = None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            u, mu, mt = (x.strip() for x in r.stdout.strip().splitlines()[0].split(","))
            v = {"util": int(float(u)), "mem_used": int(float(mu)), "mem_total": int(float(mt))}
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    _GPU_CACHE.update(t=now, v=v)
    return v


def clip_statuses():
    """Authoritative, live per-video status. Precedence (highest first):
        trained     -> its file is a source in a completed training run
        training    -> a retrain job is running now whose MKV set includes its file
        downloading -> a download job is running now for this id
        queued      -> staged in the training queue for the next retrain
        downloaded  -> file present, not yet queued or trained
        failed      -> its last download attempt failed
    Anything not listed = not downloaded."""
    idx = _load_downloads_index()
    trained = {_norm(p) for p in trained_mkvs()}
    queued_ids, queued_paths = set(), set()
    for it in _load_queue():
        if it.get("video_id"):
            queued_ids.add(it["video_id"])
        if it.get("mkv"):
            queued_paths.add(_norm(it["mkv"]))
    downloading, training_paths, failed_ids = set(), set(), set()
    for j in JOBS.values():
        cmd = j.get("cmd", "")
        if j.get("status") == "running":
            if j.get("kind") == "download":
                downloading |= set(re.findall(r"YTDL_ID=([^']+)", cmd))
            elif str(j.get("kind", "")).startswith("retrain"):
                training_paths |= {_norm(p) for p in re.findall(r"'([^']+\.mkv)'", cmd)}
        if j.get("kind") == "download" and j.get("status") == "failed":
            for it in j.get("items", []) or []:
                if it.get("status") in ("failed", "cancelled") and it.get("video_id"):
                    failed_ids.add(it["video_id"])

    out = {}
    for vid, meta in idx.items():
        n = _norm(meta["final_path"]) if meta.get("final_path") else None
        if n and n in trained:
            out[vid] = "trained"
        elif n and n in training_paths:
            out[vid] = "training"
        elif vid in queued_ids or (n and n in queued_paths):
            out[vid] = "queued"
        elif vid in downloading:
            out[vid] = "downloading"
        else:
            out[vid] = "downloaded"
    for vid in downloading:
        out.setdefault(vid, "downloading")
    for vid in failed_ids:
        out.setdefault(vid, "failed")
    return out


def build_retrain(mkvs, mode, epochs, train):
    """Both modes use Invoke-VoiceClones (auto gender-split female->s0 / male->s1;
    no reference MKV needed).
      fresh  -> train on exactly the given MKVs.
      append -> train on the deduped UNION of every previously-trained MKV + the given
                ones, so new episodes are added to the accumulated corpus, never twice."""
    scripts = cfg("scripts_dir")
    env = dict(os.environ)
    if mode == "append":
        seen, union = set(), []
        for p in list(trained_mkvs()) + list(mkvs):
            if _norm(p) not in seen:
                seen.add(_norm(p))
                union.append(p)
        mkvs = union
    if not mkvs:
        raise ValueError("no MKVs to train on")
    mkvs = [_resolve_mkv(p) for p in mkvs]
    # A missing MKV is fine when its diarization is already cached (work dir has
    # <stem>.wav + <stem>.json); training only reads the cache from then on.
    def _usable(p):
        if os.path.exists(p):
            return True
        stem = os.path.splitext(os.path.basename(p))[0]
        return (work_dir() / f"{stem}.wav").exists() and (work_dir() / f"{stem}.json").exists()
    missing = [os.path.basename(p) for p in mkvs if not _usable(p)]
    if missing:
        raise ValueError("MKV file(s) not found on disk and not diarization-cached "
                         "(moved/renamed?): " + "; ".join(missing))
    script = os.path.join(scripts, "Invoke-VoiceClones.ps1")
    cmd = f"& '{script}' -Mkvs {ps_array(mkvs)} -Epochs {int(epochs)} -Train {train}"
    argv = [cfg("powershell", "pwsh"), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd]
    return argv, cmd, env


def _available_mkvs_newest_first():
    """Every .mkv currently in download_dir, newest (most recently modified) first."""
    d = Path(cfg("download_dir"))
    mkvs = [p for p in d.glob("*.mkv") if p.is_file()] if d.exists() else []
    mkvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in mkvs]


def build_reset_retrain(epochs, train):
    """Full reset + fresh train: wipe both voices' checkpoints/datasets and the
    training history (Reset-Training.ps1), then fine-tune from the base model on
    every MKV in download_dir, newest first. Returns the newest MKV too so the
    post-run hook can pin it as the TTS reference clip."""
    scripts = cfg("scripts_dir")
    mkvs = _available_mkvs_newest_first()
    if not mkvs:
        raise ValueError(f"no .mkv files found in download_dir: {cfg('download_dir')}")
    reset = os.path.join(scripts, "Reset-Training.ps1")
    clones = os.path.join(scripts, "Invoke-VoiceClones.ps1")
    cmd = (f"$ErrorActionPreference='Stop'; "
           f"& '{reset}' -Yes -WorkDir '{cfg('work_dir')}' -VenvRoot '{cfg('venv_root')}'; "
           f"& '{clones}' -Mkvs {ps_array(mkvs)} -Epochs {int(epochs)} -Train {train}")
    argv = [cfg("powershell", "pwsh"), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd]
    return argv, cmd, dict(os.environ), mkvs[0]


def _update_tts_reference(newest_mkv):
    """Repoint both TTS speakers' reference to the newest MKV, classifying which
    diarized speaker is female/male so speaker 0/1 get the right index. Keeps the
    prior reference for a gender that can't be resolved. Returns a status note."""
    stem = Path(newest_mkv).stem
    wav, js = work_dir() / f"{stem}.wav", work_dir() / f"{stem}.json"
    if not (wav.exists() and js.exists()):
        return f"unchanged: diarization cache missing for {stem}"
    f5py = Path(cfg("venv_root")) / ".venv-f5" / "Scripts" / "python.exe"
    classify = Path(cfg("scripts_dir")) / "classify_gender.py"
    try:
        out = subprocess.run([str(f5py), str(classify), "--audio", str(wav), "--json", str(js)],
                             capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        return f"unchanged: gender classify failed to run ({e})"
    line = (out.stdout or "").strip().splitlines()[-1] if (out.stdout or "").strip() else ""
    parts = line.split()
    if len(parts) != 2:
        return f"unchanged: unexpected classify output {line!r}"
    fem, male = parts
    cfgp = HERE / "config.json"
    data = json.loads(cfgp.read_text(encoding="utf-8"))
    tts = data.get("tts") or {}
    updated = []
    for key, idx in (("0", fem), ("1", male)):
        if key in tts and idx != "-1":
            tts[key]["mkv"] = newest_mkv
            tts[key]["speaker"] = int(idx)
            updated.append(f"s{key}=SPEAKER_{int(idx):02d}")
    data["tts"] = tts
    cfgp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    CONFIG["tts"] = tts                 # keep the in-memory /api/state view current
    who = ", ".join(updated) if updated else "no gendered speaker resolved"
    return (f"set to newest clip {Path(newest_mkv).name} ({who}) — "
            "restart the TTS worker to load the new checkpoints/reference.")


def build_append(ref_mkv, ref_speaker, extra_mkvs, dataset, epochs):
    """Append extra MKVs to one existing dataset via Invoke-Finetune -AutoMatch."""
    scripts = cfg("scripts_dir")
    script = os.path.join(scripts, "Invoke-Finetune.ps1")
    cmd = (f"& '{script}' -Mkv '{ref_mkv}' -Speaker {ref_speaker} "
           f"-ExtraMkvs {ps_array(extra_mkvs)} -AutoMatch "
           f"-DatasetName {dataset} -Epochs {int(epochs)} -Force")
    argv = [cfg("powershell", "pwsh"), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd]
    return argv, cmd, dict(os.environ)


def build_download(video_ids):
    tmpl = cfg("download_cmd")
    out_dir = cfg("download_dir")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if not tmpl:
        raise ValueError("download_cmd is not configured in webui/config.json. Set it to your "
                         "downloader project's command, using {video_id} and {out_dir} placeholders.")
    # one job that downloads all selected ids sequentially; a YTDL_ID marker before
    # each call lets _run correlate the produced file path back to the video id.
    lines = []
    for v in video_ids:
        lines.append(f"Write-Output 'YTDL_ID={v}'")
        lines.append(tmpl.format(video_id=v, out_dir=out_dir))
    script = " ; ".join(lines)
    argv = [cfg("powershell", "pwsh"), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script]
    return argv, script, dict(os.environ)


# --- TTS inference (synchronous) ---------------------------------------------
def _tts_via_worker(text, speaker, out_path, speed, pause_spaces):
    """Ask the persistent tts_worker.py (models pre-loaded on GPU) to synthesize.
    Returns True on success, False if the worker isn't running / doesn't have
    this speaker (caller falls back to the PowerShell path). Raises on an actual
    synthesis error reported by the worker."""
    import socket
    port = int(cfg("tts_worker_port", 8757))
    req = {"op": "tts", "text": text, "speaker": str(speaker),
           "out_path": str(out_path), "speed": speed, "pause_spaces": pause_spaces}
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2) as s:
            s.settimeout(300)  # synthesis itself is seconds; be generous
            s.sendall(json.dumps(req).encode("utf-8") + b"\n")
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
    except OSError:
        return False  # worker not running
    resp = json.loads(buf.decode("utf-8"))
    if resp.get("ok"):
        return True
    err = resp.get("error", "")
    if "not loaded" in err:
        return False  # speaker missing in worker (e.g. ref clip not cut yet)
    raise RuntimeError(f"TTS worker error: {err}")


def tts_out_dir():
    d = work_dir() / "tts_out"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tts_speaker(speaker):
    """Resolve a speaker number to its pinned config entry, validating that the
    checkpoint and the reference source (an MKV to diarize, or a pre-cut clip) exist."""
    table = CONFIG.get("tts") or {}
    key = str(speaker)
    entry = table.get(key)
    if not entry:
        raise ValueError(f"unknown speaker {speaker!r}; valid: {sorted(table.keys())}")
    ckpt = entry.get("ckpt")
    if not ckpt or not os.path.exists(ckpt):
        raise ValueError(f"speaker {key}: ckpt not found on disk: {ckpt}")
    # Reference source: prefer anchoring to the source MKV (re-select the ref clip via
    # cached diarization each call); fall back to a pinned pre-cut ref_audio clip.
    src = entry.get("mkv") or entry.get("ref_audio")
    if not src or not os.path.exists(src):
        raise ValueError(f"speaker {key}: reference source not found on disk: {src}")
    return entry, ckpt


def synth_tts(text, speaker, out_path=None, out_dir=None, speed=None, pause_spaces=None):
    """Run F5-TTS for one utterance in the given trained voice and return the WAV path.
    Blocks until synthesis completes (a few seconds on-GPU). If out_path is given, the
    audio is written there (must be a .wav); if out_dir is given, it lands there as
    <id>.wav; otherwise it lands in work/tts_out/<id>.wav."""
    text = (text or "").strip()
    if not text:
        raise ValueError("text is empty")
    if len(text) > 2000:
        raise ValueError("text too long (max 2000 chars)")
    if speed is not None:
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            raise ValueError(f"speed must be a number, got {speed!r}")
        if not 0.3 <= speed <= 2.0:
            raise ValueError(f"speed must be between 0.3 and 2.0, got {speed}")
    if pause_spaces is not None:
        try:
            pause_spaces = int(pause_spaces)
        except (TypeError, ValueError):
            raise ValueError(f"pause_spaces must be an integer, got {pause_spaces!r}")
        if not 0 <= pause_spaces <= 10:
            raise ValueError(f"pause_spaces must be between 0 and 10, got {pause_spaces}")
    entry, ckpt = _tts_speaker(speaker)
    out_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    if not out_path and out_dir:
        out_path = Path(out_dir) / f"{out_id}.wav"
    if out_path:
        out_path = Path(out_path)
        if out_path.suffix.lower() != ".wav":
            raise ValueError("out_path must end in .wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = tts_out_dir() / f"{out_id}.wav"
    # Fast path: persistent GPU worker (tts_worker.py) with models pre-loaded.
    if _tts_via_worker(text, speaker, out_path, speed, pause_spaces):
        return out_id, out_path
    # Fallback: full Invoke-VoiceClone.ps1 pipeline (cold start, ~20-30s).
    script = os.path.join(cfg("scripts_dir"), "Invoke-VoiceClone.ps1")
    if entry.get("mkv"):
        # Anchor to the source recording: diarize (cached) + select this speaker's
        # reference clip fresh, so the reference tracks the actual #448 audio.
        spk = int(entry.get("speaker", speaker))
        ref = f"-Mkv '{entry['mkv']}' -Speaker {spk}"
    else:
        ref_text = ""
        rt = entry.get("ref_text")
        if rt and os.path.exists(rt):
            ref_text = Path(rt).read_text(encoding="utf-8").strip()
        ref = f"-RefAudio '{entry['ref_audio']}' -RefText '{ref_text}'"
    cmd = (f"& '{script}' {ref} -CkptFile '{ckpt}' -GenText '{text}' -OutFile '{out_path}'")
    if speed is not None:
        cmd += f" -Speed {speed}"
    if pause_spaces is not None:
        cmd += f" -PauseSpaces {pause_spaces}"
    argv = [cfg("powershell", "pwsh"), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd]
    logf = tts_out_dir() / f"{out_id}.log"
    with open(logf, "w", encoding="utf-8", errors="replace") as lf:
        p = subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT,
                           cwd=cfg("scripts_dir"), text=True)
    if p.returncode != 0 or not out_path.exists():
        tail = logf.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"TTS synthesis failed (exit {p.returncode}):\n{tail}")
    return out_id, out_path


def build_analyze(playlist_json=None, url=None):
    py = "python"
    script = os.path.join(cfg("scripts_dir"), "analyze_playlist.py")
    argv = [py, script, "--out", str(analysis_path())]
    if playlist_json:
        argv += ["--playlist-json", playlist_json]
    elif url:
        argv += ["--url", url]
    else:
        raise ValueError("analyze needs a playlist JSON path or URL")
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"          # titles are non-ASCII; avoid cp1252 crash on the summary print
    return argv, " ".join(argv), env


# --- HTTP --------------------------------------------------------------------
def read_json_file(p):
    p = Path(p)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quieter console
        pass

    def _send(self, obj, code=200, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # ---- GET ----
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._serve_static("index.html", "text/html; charset=utf-8")
        if path == "/api/state":
            return self._send({
                "config": {k: CONFIG.get(k) for k in
                           ("work_dir", "download_dir", "download_cmd", "default_epochs")},
                "tts_speakers": {k: v.get("label") or k
                                 for k, v in (CONFIG.get("tts") or {}).items()},
                "history": read_history(),
                "analysis": read_json_file(analysis_path()),
                "downloaded": sorted(_load_downloads_index().keys()),
                "trained_ids": _trained_video_ids(),
                "trained_count": len(trained_mkvs()),
                "clip_status": clip_statuses(),
                "training_queue": _load_queue(),
                "train_paused": TRAIN_PAUSED.is_set(),
                "training_progress": training_progress(),
                "gpu": gpu_stats(),
                "downloads": _load_downloads_index(),
                "jobs": sorted(JOBS.values(), key=lambda j: j["started"], reverse=True)[:50],
            })
        if path.startswith("/api/tts/"):
            leaf = os.path.basename(path)                 # <id>.wav ; guard traversal
            if not re.fullmatch(r"[\w.-]+\.wav", leaf):
                return self._send({"error": "bad filename"}, 400)
            f = tts_out_dir() / leaf
            if not f.exists():
                return self._send({"error": "no such audio"}, 404)
            return self._send(f.read_bytes(), ctype="audio/wav")
        if path.startswith("/api/jobs/"):
            jid = path.rsplit("/", 1)[-1]
            job = JOBS.get(jid)
            if not job:
                return self._send({"error": "no such job"}, 404)
            log = ""
            if job.get("log") and Path(job["log"]).exists():
                log = Path(job["log"]).read_text(encoding="utf-8", errors="replace")[-20000:]
            return self._send({**job, "logtext": log})
        return self._send({"error": "not found"}, 404)

    # ---- POST ----
    def do_POST(self):
        try:
            if self.path == "/api/tts":
                b = self._body()
                if "speaker" not in b:
                    return self._send({"error": "missing 'speaker'"}, 400)
                try:
                    out_id, out_path = synth_tts(b.get("text"), b.get("speaker"),
                                                 b.get("out_path"), b.get("out_dir"),
                                                 b.get("speed"), b.get("pause_spaces"))
                except ValueError as e:
                    return self._send({"error": str(e)}, 400)
                except RuntimeError as e:
                    return self._send({"error": str(e)}, 500)
                resp = {"id": out_id, "speaker": b.get("speaker"), "out_path": str(out_path)}
                # serve playback from the tts_out dir only; if the WAV was written to a
                # custom directory, keep a copy there so it can still play in the browser
                if _norm(os.path.dirname(str(out_path))) == _norm(str(tts_out_dir())):
                    resp["url"] = f"/api/tts/{os.path.basename(str(out_path))}"
                else:
                    try:
                        shutil.copyfile(out_path, tts_out_dir() / f"{out_id}.wav")
                        resp["url"] = f"/api/tts/{out_id}.wav"
                    except OSError:
                        pass
                return self._send(resp)
            if self.path == "/api/analyze":
                b = self._body()
                argv, sh, env = build_analyze(b.get("playlist_json"), b.get("url"))
                return self._send(start_job("analyze", argv, sh, env))
            if self.path == "/api/download":
                b = self._body()
                ids = [v for v in b.get("video_ids", []) if v]
                if not ids:
                    return self._send({"error": "no video_ids"}, 400)
                argv, sh, env = build_download(ids)
                return self._send(start_job("download", argv, sh, env))
            if self.path == "/api/queue":
                return self._queue(self._body())
            if self.path == "/api/train_control":
                return self._train_control(self._body())
            if self.path == "/api/retrain":
                b = self._body()
                mkvs = [m for m in b.get("mkvs", []) if m]
                if not mkvs:
                    return self._send({"error": "no mkvs"}, 400)
                argv, sh, env = build_retrain(mkvs, b.get("mode", "fresh"),
                                              b.get("epochs", cfg("default_epochs", 100)),
                                              b.get("train", "both"))
                return self._send(start_job("retrain", argv, sh, env))
            if self.path == "/api/reset_retrain":
                b = self._body()
                argv, sh, env, newest = build_reset_retrain(
                    b.get("epochs", cfg("default_epochs", 100)), b.get("train", "both"))
                return self._send(start_job("reset-retrain", argv, sh, env,
                                            meta={"newest_mkv": newest}))
            if self.path == "/api/append":
                b = self._body()
                argv, sh, env = build_append(b["ref_mkv"], b.get("ref_speaker", "0"),
                                             b.get("extra_mkvs", []),
                                             b.get("dataset", "s0"),
                                             b.get("epochs", cfg("default_epochs", 100)))
                return self._send(start_job("retrain-append", argv, sh, env))
            return self._send({"error": "not found"}, 404)
        except (ValueError, KeyError) as e:
            return self._send({"error": str(e)}, 400)

    def _train_control(self, b):
        """Control the training worker: pause / resume the queue, or stop a job.
        (Starting a training run is the existing /api/retrain, /api/reset_retrain
        and /api/queue?action=train endpoints.)"""
        action = b.get("action")
        if action == "pause":
            TRAIN_PAUSED.set()
            return self._send({"train_paused": True,
                               "note": "queue paused — the running job finishes; the next won't start"})
        if action == "resume":
            TRAIN_PAUSED.clear()
            return self._send({"train_paused": False, "note": "queue resumed"})
        if action == "stop":
            jid = b.get("id")
            if jid:
                job = JOBS.get(jid)
            else:                                   # default: the running training job
                job = next((j for j in JOBS.values()
                            if j.get("kind") in TRAINING_KINDS and j.get("status") == "running"), None)
            if not job:
                return self._send({"error": "no such job / nothing running"}, 404)
            return self._send({"id": job["id"], "note": _cancel_job(job)})
        if action == "stop_all":
            stopped = {}
            for j in list(JOBS.values()):
                if j.get("kind") in TRAINING_KINDS and j.get("status") in ("queued", "running"):
                    stopped[j["id"]] = _cancel_job(j)
            return self._send({"stopped": stopped, "count": len(stopped)})
        return self._send({"error": f"unknown train_control action '{action}'"}, 400)

    def _queue(self, b):
        """Training-queue actions: add / remove / clear / train."""
        action = b.get("action")
        items = _load_queue()
        if action == "add":
            idx = _load_downloads_index()
            add = list(b.get("items") or [])
            for vid in b.get("video_ids", []):          # resolve ids -> downloaded file
                meta = idx.get(vid)
                if meta and meta.get("final_path"):
                    add.append({"video_id": vid, "title": meta.get("title"),
                                "mkv": meta["final_path"]})
            for m in b.get("mkvs", []):                  # or raw MKV paths
                add.append({"mkv": m})
            trained = {_norm(p) for p in trained_mkvs()}
            add = [it for it in add if it.get("mkv") and _norm(it["mkv"]) not in trained]
            return self._send({"training_queue": _save_queue(items + add)})
        if action == "remove":
            key = _norm(b.get("mkv", ""))
            return self._send({"training_queue": _save_queue(
                [it for it in items if _norm(it.get("mkv", "")) != key])})
        if action == "clear":
            return self._send({"training_queue": _save_queue([])})
        if action == "train":
            mkvs = [it["mkv"] for it in items if it.get("mkv")]
            if not mkvs:
                return self._send({"error": "training queue is empty"}, 400)
            argv, sh, env = build_retrain(mkvs, b.get("mode", "append"),
                                          b.get("epochs", cfg("default_epochs", 100)),
                                          b.get("train", "both"))
            return self._send(start_job("retrain", argv, sh, env))
        return self._send({"error": f"unknown queue action '{action}'"}, 400)

    def _serve_static(self, name, ctype):
        f = STATIC / name
        if not f.exists():
            return self._send({"error": f"missing {name}"}, 404)
        self._send(f.read_bytes(), ctype=ctype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8756)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    _load_jobs()
    threading.Thread(target=_train_worker, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Voice-clone dashboard: http://{args.host}:{args.port}")
    print(f"  history : {history_path()}")
    print(f"  analysis: {analysis_path()}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
