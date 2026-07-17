"""Persistent F5-TTS inference worker for the dashboard.

Runs under the .venv-f5 python (NOT the stdlib-only dashboard venv):

  C:/Users/miffy/Documents/whisperX/.venv-f5/Scripts/python.exe webui/tts_worker.py

At startup it loads the vocoder once plus one model per speaker checkpoint in
config.json["tts"], preprocesses each speaker's reference clip once, then serves
newline-delimited JSON over TCP on 127.0.0.1:8757. This removes the per-request
process/model-load cost (~20-30s) of the Invoke-VoiceClone.ps1 path; server.py
falls back to that path automatically when this worker is not running.

Reference clips are the ones Invoke-VoiceClone.ps1 already writes to the work
dir ({mkv-stem}_speaker{N}_ref.wav + .txt). If a speaker's clip is missing, that
speaker is skipped here (the fallback path will create it); restart the worker
to pick it up. Restart the worker after retraining to load new checkpoints.

Protocol (one JSON object per line, response is one JSON line):
  {"op": "ping"}                                    -> {"ok": true, "speakers": [...]}
  {"op": "tts", "text": ..., "speaker": "0",
   "out_path": ..., "speed": 1.0, "pause_spaces": 2} -> {"ok": true, "out_path": ...}
                                                       or {"ok": false, "error": ...}
"""
import argparse
import json
import os
import re
import socketserver
import sys
import threading
import traceback
from importlib.resources import files
from pathlib import Path

# All models (vocoder, whisper for ref auto-transcribe) are already in the HF
# cache after any prior run; skip hub network checks at startup. Set
# TTS_WORKER_ONLINE=1 to allow downloads on a fresh install.
if not os.environ.get("TTS_WORKER_ONLINE"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import soundfile as sf
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))

# Keep torch.compile's kernel cache in a stable location so worker restarts
# warm up in seconds instead of recompiling (~90s per model on a cold cache).
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                      str(Path(CONFIG["work_dir"]) / "inductor_cache"))

F5_MODEL = "F5TTS_v1_Base"
NFE_STEP = 32          # match Invoke-VoiceClone.ps1 defaults
CFG_STRENGTH = 2.0
TARGET_RMS = 0.1
PAUSE_SPACES = 2
PAD_TAIL_SEC = 0.35    # trailing silence, matches the ffmpeg apad in the script

SPEAKERS = {}          # key -> {"model", "ref_file", "ref_text", "label"}
VOCODER = None
INFER_LOCK = threading.Lock()


def log(*a):
    print(*a, flush=True)


def add_terminal_punct(t):
    """Port of Add-TerminalPunct: F5 budgets duration from text; text without
    terminal punctuation gets its last syllable clipped."""
    t = t.strip()
    if t and t[-1] not in "。．.!！?？;；":
        t += "。" if re.search(r"[一-鿿]$", t) else "."
    return t + " "


def prep_gen_text(text, pause_spaces=None):
    text = add_terminal_punct(text)
    n = PAUSE_SPACES if pause_spaces is None else int(pause_spaces)
    if n > 0:
        text = re.sub(r"([，,、；;])\s*", r"\1" + " " * n, text)
    return text


def resolve_ref(entry, key):
    """Locate the speaker's reference clip + transcript the same way
    Invoke-VoiceClone.ps1 lays them out in the work dir."""
    if entry.get("mkv"):
        base = Path(entry["mkv"]).stem
        spk = int(entry.get("speaker", key))
        ref = Path(CONFIG["work_dir"]) / f"{base}_speaker{spk}_ref.wav"
    elif entry.get("ref_audio"):
        ref = Path(entry["ref_audio"])
    else:
        raise FileNotFoundError(f"speaker {key}: no mkv/ref_audio in config")
    if not ref.exists():
        raise FileNotFoundError(f"speaker {key}: reference clip not found: {ref} "
                                "(run one synthesis via the fallback path to create it)")
    txt = Path(str(ref) + ".txt")
    if entry.get("ref_text") and Path(entry["ref_text"]).exists():
        ref_text = Path(entry["ref_text"]).read_text(encoding="utf-8").strip()
    elif txt.exists():
        ref_text = txt.read_text(encoding="utf-8").strip()
    else:
        ref_text = ""  # preprocess_ref_audio_text will auto-transcribe once
    if ref_text:
        ref_text = add_terminal_punct(ref_text)
    return str(ref), ref_text


def compile_and_warmup(key, spk):
    """torch.compile the DiT (~2.4x faster steps: ~1.6s -> ~0.7s per utterance),
    then warm it up with two different-length texts so both the initial compile
    and the dynamic-shape recompile happen here, never on a user request.
    First-ever compile is slow (~90s/model); later restarts hit the on-disk
    inductor cache and warm up in seconds. Falls back to eager on any failure.
    Disable with TTS_WORKER_NO_COMPILE=1."""
    import time
    import torch
    eager = spk["model"].transformer
    try:
        spk["model"].transformer = torch.compile(eager, dynamic=True)
        for text in ("早安。 ", "今天天氣很好，我們決定去公園散步，順便買一杯咖啡。 "):
            t0 = time.time()
            infer_process(spk["ref_file"], spk["ref_text"], text,
                          spk["model"], VOCODER, nfe_step=NFE_STEP,
                          cfg_strength=CFG_STRENGTH, target_rms=TARGET_RMS,
                          show_info=lambda *a: None)
            log(f"[worker] speaker {key}: warmup pass {time.time() - t0:.1f}s")
    except Exception as e:
        log(f"[worker] WARN: torch.compile failed for speaker {key}; using eager: {e}")
        spk["model"].transformer = eager


def load_speakers():
    global VOCODER
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{F5_MODEL}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    mel_type = model_cfg.model.mel_spec.mel_spec_type

    log(f"[worker] loading vocoder ({mel_type})...")
    VOCODER = load_vocoder(mel_type)

    for key, entry in (CONFIG.get("tts") or {}).items():
        try:
            ckpt = entry["ckpt"]
            if not Path(ckpt).exists():
                raise FileNotFoundError(f"ckpt not found: {ckpt}")
            ref_file, ref_text = resolve_ref(entry, key)
            log(f"[worker] speaker {key} ({entry.get('label', '')}): loading {ckpt}")
            model = load_model(model_cls, model_cfg.model.arch, ckpt, mel_type)
            # Clip/normalize the reference once; auto-transcribe here if no text.
            ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text, show_info=log)
            SPEAKERS[key] = {"model": model, "ref_file": ref_file,
                             "ref_text": ref_text, "label": entry.get("label", "")}
            if not os.environ.get("TTS_WORKER_NO_COMPILE"):
                compile_and_warmup(key, SPEAKERS[key])
        except Exception as e:
            log(f"[worker] WARN: speaker {key} unavailable: {e}")
    if not SPEAKERS:
        log("[worker] ERROR: no speakers loaded; exiting.")
        sys.exit(1)
    log(f"[worker] ready; speakers: {sorted(SPEAKERS)}")


def synthesize(req):
    text = (req.get("text") or "").strip()
    if not text:
        raise ValueError("text is empty")
    key = str(req.get("speaker"))
    spk = SPEAKERS.get(key)
    if not spk:
        raise ValueError(f"speaker {key!r} not loaded; available: {sorted(SPEAKERS)}")
    speed = float(req.get("speed") or 1.0)
    out_path = Path(req["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gen_text = prep_gen_text(text, req.get("pause_spaces"))
    with INFER_LOCK:
        wav, sr, _ = infer_process(
            spk["ref_file"], spk["ref_text"], gen_text,
            spk["model"], VOCODER,
            nfe_step=NFE_STEP, cfg_strength=CFG_STRENGTH,
            target_rms=TARGET_RMS, speed=speed, show_info=log,
        )
    # Trailing silence pad: F5 output often ends within ms of the last syllable.
    wav = np.concatenate([wav, np.zeros(int(sr * PAD_TAIL_SEC), dtype=wav.dtype)])
    sf.write(str(out_path), wav, sr)
    return str(out_path)


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        line = self.rfile.readline()
        if not line:
            return
        try:
            req = json.loads(line.decode("utf-8"))
            if req.get("op") == "ping":
                resp = {"ok": True, "speakers": sorted(SPEAKERS)}
            elif req.get("op") == "tts":
                resp = {"ok": True, "out_path": synthesize(req)}
            else:
                resp = {"ok": False, "error": f"unknown op {req.get('op')!r}"}
        except Exception as e:
            log("[worker] request failed:\n" + traceback.format_exc())
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self.wfile.write(json.dumps(resp).encode("utf-8") + b"\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int,
                    default=int(CONFIG.get("tts_worker_port") or 8757))
    args = ap.parse_args()

    load_speakers()
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", args.port), Handler)
    srv.daemon_threads = True
    log(f"[worker] listening on 127.0.0.1:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
