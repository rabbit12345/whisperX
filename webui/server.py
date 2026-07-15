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
import shlex
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


# --- job runner --------------------------------------------------------------
JOBS = {}          # id -> {id, kind, cmd, status, started, ended, log}
JOBS_LOCK = threading.Lock()


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
            job["status"] = "done" if rc == 0 else "failed"
            job["returncode"] = rc
        except Exception as e:  # noqa: BLE001
            lf.write(f"\n[launcher error] {e}\n")
            job["status"] = "failed"
            job["returncode"] = -1
    # For download jobs the downloader names files by title, so capture the exact
    # produced MKV path from each `--json` result line (keys: status, final_path).
    if job["kind"] == "download":
        produced = []
        for line in logf.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    d = json.loads(line)
                    if d.get("status") == "done" and d.get("final_path"):
                        produced.append(d["final_path"])
                except ValueError:
                    pass
        job["produced"] = produced
    job["ended"] = datetime.now().isoformat(timespec="seconds")
    _persist_jobs()


def start_job(kind, argv, shell_cmd=None, env=None):
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id, "kind": kind, "cmd": shell_cmd or " ".join(argv),
            "status": "running", "started": datetime.now().isoformat(timespec="seconds"),
            "ended": None, "returncode": None,
        }
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
                # anything left "running" from a previous server run is stale
                if j.get("status") == "running":
                    j["status"] = "unknown"
                JOBS[j["id"]] = j
        except (OSError, ValueError):
            pass


# --- pipeline command builders -----------------------------------------------
def ps_array(items):
    """Render a PowerShell string[] literal with each item single-quoted."""
    return "@(" + ",".join("'" + str(i).replace("'", "''") + "'" for i in items) + ")"


def build_retrain(mkvs, mode, epochs, train):
    """mode: 'fresh' -> Invoke-VoiceClones (rebuild both voices from these MKVs)
             'append'-> Invoke-Finetune -AutoMatch (add MKVs to existing s0/s1)."""
    scripts = cfg("scripts_dir")
    env = dict(os.environ)
    if mode == "append":
        # Requires a reference recording already diarized; append to each dataset.
        # We append the same new MKVs to whichever dataset(s) are requested using
        # the existing -Sources flow is per-speaker; -AutoMatch needs a reference.
        raise ValueError("append mode requires a reference MKV+speaker per voice; "
                         "use the retrain form's per-voice reference fields (see UI).")
    script = os.path.join(scripts, "Invoke-VoiceClones.ps1")
    cmd = f"& '{script}' -Mkvs {ps_array(mkvs)} -Epochs {int(epochs)} -Train {train}"
    argv = [cfg("powershell", "pwsh"), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd]
    return argv, cmd, env


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
    # one job that downloads all selected ids sequentially
    lines = [tmpl.format(video_id=v, out_dir=out_dir) for v in video_ids]
    script = " ; ".join(lines)
    argv = [cfg("powershell", "pwsh"), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script]
    return argv, script, dict(os.environ)


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
                "history": read_json_file(history_path()) or [],
                "analysis": read_json_file(analysis_path()),
                "jobs": sorted(JOBS.values(), key=lambda j: j["started"], reverse=True)[:50],
            })
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
            if self.path == "/api/retrain":
                b = self._body()
                mkvs = [m for m in b.get("mkvs", []) if m]
                if not mkvs:
                    return self._send({"error": "no mkvs"}, 400)
                argv, sh, env = build_retrain(mkvs, b.get("mode", "fresh"),
                                              b.get("epochs", cfg("default_epochs", 100)),
                                              b.get("train", "both"))
                return self._send(start_job("retrain", argv, sh, env))
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
