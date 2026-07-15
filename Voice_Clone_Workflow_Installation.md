# Voice Clone Workflow — Installation Guide

Fully local pipeline on Windows 11: take an `.mkv`, isolate one speaker's speech,
and clone that voice for text-to-speech with **F5-TTS** (zero-shot — no training step).

Pipeline: `MKV → extract audio (FFmpeg) → transcribe + diarize (WhisperX) → cut speaker clips (FFmpeg) → reference clip → F5-TTS synthesis`

## One-stop script (after install)

Once the environments below are set up, the whole pipeline runs from one command
via `scripts\Invoke-VoiceClone.ps1` (it calls `scripts\select_reference.py`):

```powershell
# See who's talking first (diarize only):
.\scripts\Invoke-VoiceClone.ps1 -Mkv "X:\media\clip.mkv" -HfToken $env:HF_TOKEN -ListSpeakers

# Then clone the chosen speaker saying your text:
.\scripts\Invoke-VoiceClone.ps1 -Mkv "X:\media\clip.mkv" -HfToken $env:HF_TOKEN `
    -Speaker 0 -GenText "你好，这是克隆的声音。" -OutFile clone.wav
```

Output lands in `%USERPROFILE%\Documents\whisperX\work\`. Diarization (the slow
step) is cached per source file, so re-running with new `-GenText` is fast; pass
`-Force` to redo it.

**Quality knobs** (all optional): `-NfeStep 48` (more denoising steps — cleaner,
slower), `-CfgStrength 2.5` (hug the reference more), `-RemoveSilence`,
`-CleanRef` (gentle highpass + denoise + loudnorm on the reference clip),
`-Vocoder bigvgan`, `-TargetRms`, `-Speed`. Reference text defaults to the
WhisperX transcript; override with `-RefText "..."` or `-AutoRefText` (let F5
transcribe the clip). Example:

```powershell
.\scripts\Invoke-VoiceClone.ps1 -Mkv "X:\media\clip.mkv" -Speaker 0 `
    -GenText "你想合成的文字。" -OutFile clone.wav `
    -CleanRef -NfeStep 48 -RemoveSilence
```
 The script auto-selects the target speaker's **longest
clean 8–12 s reference clip** (longer = better clone quality) and uses its
transcript as the reference text. F5-TTS hard-clips the reference at 12 s, so
`-MaxDur` should not exceed 12; tune with `-MinDur`/`-MaxDur` if needed. The rest
of this document covers the one-time installation the script depends on.

## Running offline

After the one-time model downloads (transcription, zh alignment, pyannote
diarization, F5-TTS + vocab, Vocos vocoder — all cached under
`~/.cache/huggingface`), the whole pipeline is local compute. Add `-Offline` to
either script to force no-network mode (sets `HF_HUB_OFFLINE` /
`TRANSFORMERS_OFFLINE` / `HF_DATASETS_OFFLINE`, and points `--pretrain` at the
cached base checkpoint for fine-tuning):

```powershell
.\scripts\Invoke-VoiceClone.ps1 -Mkv "X:\media\clip.mkv" -Speaker 0 -GenText "..." -Offline
.\scripts\Invoke-Finetune.ps1   -Mkv "X:\media\clip.mkv" -Speaker 0 -DatasetName s0 -Offline
```

The only steps that ever need internet are the **first** transcription,
diarization, and synthesis (to fetch + cache models) and accepting the pyannote
gated-model terms. Verified: inference and fine-tuning both run with all HF
offline flags set.

## Two voices from many recordings — one command

`scripts\Invoke-VoiceClones.ps1` is the streamlined all-in-one: supply every
recording, and it diarizes each, **auto-detects which speaker is female vs male
by pitch** (female → `s0`, male → `s1`), combines all female audio into one
dataset and all male into another, and fine-tunes both.

```powershell
$env:HF_TOKEN = "hf_..."
# Preview the gender assignment first (diarizes, no training):
.\scripts\Invoke-VoiceClones.ps1 -Mkvs @("C:\...\ep446.mkv","C:\...\ep448.mkv","C:\...\ep450.mkv") -NoTrain

# Then run the full pipeline (diarize all + fine-tune s0=female, s1=male):
.\scripts\Invoke-VoiceClones.ps1 -Mkvs @("C:\...\ep446.mkv","C:\...\ep448.mkv","C:\...\ep450.mkv") -Epochs 100
```

Options: `-Train female|male|both` (default both), `-Offline`,
`-FemaleDataset`/`-MaleDataset` (default `s0`/`s1`). It reuses the tested
diarize (`Invoke-VoiceClone -ListSpeakers`), gender (`classify_gender.py`),
combine (`build_finetune_dataset.py --append`), and train (`Invoke-Finetune`)
pieces, and rebuilds each dataset from exactly the MKVs you supply.

**Assumptions (gender-grouping mode):** each recording has exactly one female +
one male speaker, and the supplied recordings share the **same two hosts** — it
pools *all* female audio into `s0` and *all* male into `s1` by pitch, so mixing
different shows would blend different people. Only feed it same-host episodes.
If a file's F0 gap is < 20 Hz it warns (ambiguous). Inference then uses the
fine-tuned checkpoint via `-CkptFile` (see below). Diarization + training use the
GPU — run when it's free.

## Fine-tuning a single dedicated voice (lower-level)

Zero-shot cloning mimics a 12 s clip; **fine-tuning** trains a speaker-specific
checkpoint for better fidelity. `scripts\Invoke-Finetune.ps1` runs the whole
flow off the cached diarization:

```powershell
# Requires a prior Invoke-VoiceClone run (so <name>.json exists).
.\scripts\Invoke-Finetune.ps1 -Mkv "X:\media\clip.mkv" -Speaker 0 -DatasetName s0 -Epochs 100
```

It (1) re-chunks the speaker's word-level timestamps into clean 3–12 s clips
(`build_finetune_dataset.py` — uses the *whole* speaker track, not just short
segments), (2) builds the arrow dataset via `prepare_csv_wavs` (omit
`--pretrain` = fine-tune mode, reuses the base pinyin vocab), and (3) fine-tunes
`F5TTS_v1_Base`. Checkpoints (with `--log_samples` audio previews) land in
`<venv>\Lib\ckpts\<DatasetName>\`. Then infer with the fine-tuned model:

```powershell
.\scripts\Invoke-VoiceClone.ps1 -Mkv "X:\media\clip.mkv" -Speaker 0 `
    -GenText "你想合成的文字。" -CkptFile "C:\...\.venv-f5\Lib\ckpts\s0\model_last.pt"
```

**Combining multiple recordings** (more data = better quality). This is now a
**single command** — any recording not yet diarized is auto-diarized first (it
calls `Invoke-VoiceClone -ListSpeakers` internally, so pass `-HfToken`), then the
target speaker is matched per file and everything is combined and trained:

```powershell
$env:HF_TOKEN = "hf_..."
# Auto-match: -Mkv/-Speaker is the reference voice; labels in -ExtraMkvs are auto-detected
# (via the cached wespeaker embedding model, CPU-only), and any uncached file is auto-diarized.
.\scripts\Invoke-Finetune.ps1 -DatasetName hostA -Mkv "C:\...\ep448.mkv" -Speaker 0 `
    -ExtraMkvs @("C:\...\ep450.mkv","C:\...\ep451.mkv") -AutoMatch

# Or specify labels manually (skips auto-match) with -Sources:
.\scripts\Invoke-Finetune.ps1 -DatasetName hostA -Sources @(
    "C:\...\ep448.mkv|0", "C:\...\ep450.mkv|1", "C:\...\ep451.mkv|0")
```

Pipeline internals: `[0/3]` ensures diarization for every recording,
`match_speaker.py` prints the matched label per file, and
`build_finetune_dataset.py --append` accumulates all clips into one dataset.
(Diarization + training both use the GPU, so run when it's free.)

Notes: ~15 min of speaker audio is the low end (helps, but 30+ min is better).
Data/ckpts live under the venv's `Lib\` (F5-TTS's fixed layout). The base pinyin
vocab is auto-placed at `data\Emilia_ZH_EN_pinyin\vocab.txt` on first run.
Delete `ckpts\<name>` to restart training from the base model.

## Environment

| Item        | Value                                            |
|-------------|--------------------------------------------------|
| OS          | Windows 11                                        |
| GPU         | RTX 5090 (Blackwell, `sm_120`), 32 GB VRAM        |
| CUDA        | Latest toolkit + current NVIDIA driver installed  |
| Python      | 3.11 (`py -3.11`)                                 |
| Source      | `.mkv`                                            |
| Consent     | Obtained                                          |

> **Blackwell requirement.** The RTX 5090 needs PyTorch built for **CUDA 12.8**
> (`cu128` wheels). Older `cu121`/`cu124` wheels will not use the GPU. All
> installs below pin the `cu128` index.
>
> Your **system CUDA toolkit version does not need to match** — the `cu128`
> wheels bundle their own CUDA 12.8 runtime. Only the NVIDIA **driver** matters,
> and it must be recent enough to run that runtime (driver 596.49 / CUDA 13.2
> toolkit is fine; newer drivers are backward-compatible with older runtimes).

> **Two virtual environments.** WhisperX and F5-TTS pin conflicting versions of
> `torch`/`transformers`/`numpy`. Install each in its own venv. Never share one.

## Prerequisites

- NVIDIA driver current; `nvidia-smi` runs and shows the 5090.
- ~30 GB free disk (models + intermediate WAVs).
- Hugging Face account + access token (for WhisperX diarization).
- Python 3.11 installed (`winget install Python.Python.3.11`).

## 0. Folder layout

```powershell
$root = "voice-clone-workflow"
mkdir $root\input, $root\audio, $root\transcripts, $root\profiles, $root\output, $root\scripts
```

```text
voice-clone-workflow/
├── input/          # source.mkv (copy; keep original untouched)
├── audio/          # extracted + speaker clips
├── transcripts/    # WhisperX output (json/srt)
├── profiles/       # per-speaker reference.wav + metadata
├── output/         # F5-TTS generated speech
└── scripts/
```

## 1. FFmpeg

```powershell
winget install Gyan.FFmpeg
# open a NEW terminal, then verify:
ffmpeg -version
```

## 2. WhisperX (transcription + diarization)

Create the venv on a **local disk** (not a network share — SMB corrupts
torch-sized installs):

```powershell
py -3.11 -m venv C:\Users\miffy\Documents\whisperX\.venv-whisperx
C:\Users\miffy\Documents\whisperX\.venv-whisperx\Scripts\Activate.ps1

# Install WhisperX FIRST — it pins torch and will pull a CPU build if torch
# is already present.
pip install whisperx

# Then force the matching torch version from the Blackwell (CUDA 12.8) index.
# WhisperX currently pins torch 2.8.0.
pip uninstall -y torch torchaudio
pip install "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128

# Verify GPU + WhisperX together — must print +cu128 and True
python -c "import torch, whisperx; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> 2.8.0+cu128 True NVIDIA GeForce RTX 5090
```

**WhisperX uses torchcodec too, and needs FFmpeg 7 (not 8).** WhisperX pins
torchcodec 0.7.0 (for torch 2.8.0), which supports FFmpeg 4–7 only — so the
FFmpeg 8 shared build the F5-TTS venv uses will not load here. Install the
FFmpeg 7 shared build and add a `sitecustomize.py` to this venv keyed on
`avcodec-61.dll`:

```powershell
winget install -e --id BtbN.FFmpeg.LGPL.Shared.7.1
# sitecustomize.py globs %LOCALAPPDATA%\...\Packages\*\*\bin\avcodec-61.dll and
# calls os.add_dll_directory (this repo ships a copy under the whisperx venv).
python -c "import torch; from torchcodec.decoders import AudioDecoder; import whisperx; print('OK', torch.cuda.is_available())"
# -> OK True
```

> Each venv points at a different FFmpeg: **whisperx → FFmpeg 7** (torchcodec
> 0.7, `avcodec-61.dll`), **f5 → FFmpeg 8** (torchcodec 0.12, `avcodec-62.dll`).

One-time: accept the model terms on Hugging Face for
`pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`, then create a
read token at https://huggingface.co/settings/tokens.

## 3. Extract audio from the MKV

Mono, 24 kHz WAV — good for both WhisperX and as F5-TTS reference source.

```powershell
copy your_video.mkv input\source.mkv
ffmpeg -i input\source.mkv -vn -ac 1 -ar 24000 -c:a pcm_s16le audio\source.wav
```

## 4. Transcribe + identify speakers

```powershell
whisperx audio\source.wav `
  --model large-v3 --language zh `
  --diarize --hf_token YOUR_HF_TOKEN `
  --compute_type float16 `
  --output_format json --output_dir transcripts
```

`large-v3` transcribes Mandarin well; `--language zh` skips auto-detection.
The `text` fields in the JSON will be Chinese characters — copy them verbatim
for the reference text in step 7.

Output `transcripts\source.json` contains segments with `start`, `end`, `text`,
and a `speaker` label (e.g. `SPEAKER_00`). Skim it to pick the target speaker.

## 5. Cut a clean reference clip

F5-TTS needs **one clean 5–15 s clip** of the target speaker (single voice, no
music/crosstalk). Find a good segment in the JSON, then cut it:

```powershell
mkdir profiles\Speaker_0
# replace START/END with seconds from the chosen segment
ffmpeg -i audio\source.wav -ss START -to END -ac 1 -ar 24000 -c:a pcm_s16le profiles\Speaker_0\reference.wav
```

Copy that segment's `text` from the JSON — you'll pass it as the reference text.

## 6. F5-TTS (voice cloning)

`F5TTS_v1_Base` is **bilingual (Mandarin + English)** — no separate Chinese
model is required. Reference and generated text should be Chinese characters.

Separate venv on a **local disk**. F5-TTS leaves torch unpinned, so it pulls a
CPU build (and a torch newer than the `cu128` index carries). Install F5-TTS,
then pin the whole torch trio to `2.11.0+cu128` — the newest the `cu128` index
has:

```powershell
py -3.11 -m venv C:\Users\miffy\Documents\whisperX\.venv-f5
C:\Users\miffy\Documents\whisperX\.venv-f5\Scripts\Activate.ps1
python -m pip install -U pip
pip install f5-tts

# Pin CUDA-enabled torch + a torchcodec built for that torch (0.12 ↔ 2.11)
pip install --force-reinstall --no-deps "torch==2.11.0" "torchaudio==2.11.0" --index-url https://download.pytorch.org/whl/cu128
pip install --no-deps "torchcodec==0.12.0"
```

**F5-TTS needs FFmpeg *shared* DLLs (torchcodec dependency).** The static
`Gyan.FFmpeg` build has no DLLs, so also install the shared build:

```powershell
winget install -e --id Gyan.FFmpeg.Shared
```

Windows (Python 3.8+) ignores `PATH` when resolving a DLL's dependencies, so
torchcodec can't find the FFmpeg DLLs on its own. Add a `sitecustomize.py` to
the venv's `site-packages` that registers the FFmpeg `bin` at startup (this repo
ships a copy — it globs `%LOCALAPPDATA%\...\Gyan.FFmpeg.Shared*\*\bin` and calls
`os.add_dll_directory`). Verify:

```powershell
python -c "import torch; from torchcodec.decoders import AudioDecoder; print(torch.__version__, torch.cuda.is_available())"
# -> 2.11.0+cu128 True
```

## 7. Synthesize with the cloned voice

**Direct synthesis from a reference clip** (skips diarization; use a fine-tuned
voice via `-CkptFile`):

```powershell
.\scripts\Invoke-VoiceClone.ps1 -RefAudio "C:\...\work\<name>_speaker0_ref.wav" `
    -GenText "你想合成的文字。" -CkptFile "C:\...\.venv-f5\Lib\ckpts\s0\model_last.pt" `
    -OutFile clone.wav
```

`-RefAudio` reads the clip's `.txt` sidecar for the reference text (or use
`-RefText`/`-AutoRefText`). Omit `-CkptFile` for the base zero-shot model. Use a
female clip with `s0`, a male clip with `s1`. Quality knobs (`-NfeStep 48`,
`-RemoveSilence`, `-CleanRef`, `-CfgStrength`) all apply.

Lower-level F5-TTS CLI: set `$env:PYTHONUTF8=1` first — otherwise it crashes with
`UnicodeEncodeError: 'charmap'` when it prints Chinese to the console. And pass
an **absolute** `--ref_audio` path: a relative path is resolved against the
F5-TTS package dir and won't be found.

```powershell
$env:PYTHONUTF8 = "1"
f5-tts_infer-cli `
  --model F5TTS_v1_Base `
  --ref_audio C:\Users\miffy\Documents\whisperX\profiles\Speaker_0\reference.wav `
  --ref_text "参考音频的逐字转录" `
  --gen_text "想要用克隆声音说出的文字。" `
  --output_dir output --output_file clone.wav
```

Model weights (~1.3 GB) download automatically on first run. Output is 24 kHz
mono WAV in `output\`. (Leave `--ref_text ""` to let F5-TTS auto-transcribe the
reference instead of supplying it.)

Verified working: an 8 s Mandarin clip generates in ~3 s on the RTX 5090.

## Profile layout (reusable voices)

```text
profiles/Speaker_0/
├── reference.wav
└── metadata.json
```

```json
{
  "display_name": "TODO",
  "source_recording": "input/source.mkv",
  "reference_text": "exact transcript of reference.wav",
  "consent_status": "obtained",
  "language": "zh",
  "date_created": "TODO"
}
```

## Reference-clip quality checklist

Good clips are single-speaker, low-noise, stable volume, natural tone, 5–15 s.
Avoid laughter, crosstalk, music bleed, distortion, and phone-quality audio.

## Troubleshooting (RTX 5090 / Blackwell)

- **`torch.cuda.is_available()` is `False`, version shows `+cpu`** — a plain
  `pip install torch` on Windows (or letting WhisperX pull torch in) grabs the
  CPU-only wheel. Fix:
  `pip uninstall -y torch torchaudio` then reinstall from the `cu128` index above.
  Verify the version prints `+cu128`, not `+cpu`. Installing WhisperX can
  overwrite torch with the CPU build, so run this **after** `pip install whisperx`.
- **CTranslate2 / faster-whisper errors or CPU fallback** — WhisperX's backend
  needs a build that supports `sm_120`: `pip install -U ctranslate2 faster-whisper`.
- **Diarization fails / 401** — accept the pyannote model terms on Hugging Face
  and confirm the token is a valid **read** token.
- **Out of memory** — unlikely at 32 GB, but drop `--model large-v3` to
  `medium`, or add `--batch_size 8` to the WhisperX command.
```