# Voice-Clone Dashboard — Instructions

Local dashboard + backend for the voice-clone training pipeline (`webui/`).
Everything runs locally; the server is Python stdlib only (no pip installs needed).

## Start / restart / stop the server

Use `Start-Server.ps1` (PowerShell, from the repo root):

```powershell
.\Start-Server.ps1          # start if not running; restart if already running
.\Start-Server.ps1 -Stop    # stop the server
.\Start-Server.ps1 -Port 9000   # use a different port (default 8756)
```

Notes:
- Detection is port-based (`Get-NetTCPConnection` on the port), so it finds the
  server no matter how it was originally launched.
- The server runs hidden; output goes to `webui/server.log` and
  `webui/server.log.err`. If startup fails, the script prints the last error lines.
- Manual alternative: `python webui/server.py [--port 8756]`

Then open **http://127.0.0.1:8756**

## Configuration (`webui/config.json`)

| Key | Purpose |
|---|---|
| `work_dir` | Working directory (history, jobs, TTS output live here) |
| `download_dir` / `download_cmd` | Where MKVs land and the downloader command template (`{video_id}`, `{out_dir}`) |
| `scripts_dir` | Location of the pipeline scripts (`Invoke-VoiceClones.ps1`, etc.) |
| `history_file` / `analysis_file` | Override paths for training history / playlist analysis |
| `hf_token_env` | Env var holding the HF token (default `HF_TOKEN`) |
| `default_epochs` | Default epoch count for retraining |
| `powershell` | PowerShell executable (default `pwsh`) |
| `tts` | Per-speaker TTS entries: `label`, `ckpt` (trained checkpoint), `mkv` + `speaker` (reference source, re-selected via cached diarization) or `ref_audio`/`ref_text` (pinned clip) |

## Dashboard actions

- **Analyze** — classify a playlist (`scripts/analyze_playlist.py`); results feed the guest-filter view.
- **Download** — fetch selected video ids via `download_cmd` (the YtubeDownloader project).
- **Retrain** — fresh training via `Invoke-VoiceClones.ps1`, or **Append** to add new MKVs to the running s0/s1 datasets (`Invoke-Finetune.ps1 -AutoMatch`).
- **Reset & retrain from scratch** — wipes both voices' checkpoints, prepared/staged datasets and clears training history (`Reset-Training.ps1`), then fine-tunes from the base model on every MKV in `download_dir` (newest first) via `Invoke-VoiceClones.ps1`. The newest clip is pinned as the TTS reference in `config.json`; restart the TTS worker afterward to load the new checkpoints/reference.
- **TTS** — synthesize speech in a trained voice (female `s0` / male `s1`). Runs `Invoke-VoiceClone.ps1` synchronously (a few seconds on GPU); output WAVs land in `<work_dir>/tts_out/`.

## API (for scripting)

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/state` | GET | Config summary, training history, job/clip status, TTS speakers |
| `/api/tts` | POST | `{"text": "...", "speaker": 0}` → synthesized WAV (max 2000 chars); response includes a `/api/tts/<id>.wav` URL |
| `/api/tts/<file>.wav` | GET | Fetch a synthesized WAV |
| `/api/analyze` | POST | Run playlist analysis |
| `/api/download` | POST | Download videos by id |
| `/api/queue` | POST | Queue training jobs |
| `/api/retrain` | POST | Fresh retrain |
| `/api/reset_retrain` | POST | Reset all training state, then retrain from scratch on every MKV in `download_dir` |
| `/api/append` | POST | Append new MKVs to existing datasets |
| `/api/jobs/<id>` | GET | Background job status/log |

Example TTS call:

```powershell
Invoke-RestMethod http://127.0.0.1:8756/api/tts -Method Post -ContentType 'application/json' `
  -Body (@{ text = '你好，世界'; speaker = 0 } | ConvertTo-Json)
```

## Troubleshooting

- **Server won't start** — check `webui/server.log.err`; commonly a bad path in `config.json`.
- **TTS fails** — verify the `ckpt` and `mkv`/`ref_audio` paths in the `tts` config entry exist; per-synthesis logs are in `<work_dir>/tts_out/<id>.log`.
- **Port already in use by something else** — pass a different `-Port` (and `--port` matches on the Python side).
