<#
.SYNOPSIS
  Automated download + install + setup for the whisperX + F5-TTS voice-clone
  workflow on Windows (RTX 5090 / Blackwell, CUDA 12.8).

.DESCRIPTION
  Prompts for an install location, then reproduces the whole hand-validated
  environment:
    * two Python 3.11 venvs (.venv-whisperx, .venv-f5) on a LOCAL disk
    * pinned cu128 torch + matching torchcodec in each
    * FFmpeg 7 (shared) for whisperx, FFmpeg 8 (shared) for f5
    * the two sitecustomize.py DLL shims that let torchcodec find the FFmpeg DLLs
    * base pinyin vocab placement for fine-tuning
    * webui/config.json wired to the chosen paths
  Optionally restores a data backup (trained models + datasets + caches).

.EXAMPLE
  # Interactive (prompts for location + HF token):
  .\Install-VoiceCloneWorkflow.ps1

.EXAMPLE
  # Unattended, explicit location, restore a backup afterwards:
  .\Install-VoiceCloneWorkflow.ps1 -InstallRoot C:\voice -HfToken hf_xxx `
      -RestoreFrom D:\backups\voice-2026-07-16 -Yes
#>
[CmdletBinding()]
param(
  [string]$InstallRoot,                       # venv + data root (LOCAL disk). Prompted if omitted.
  [string]$HfToken = $env:HF_TOKEN,           # optional; persists HF_TOKEN for diarization
  [string]$RestoreFrom,                        # optional backup folder to restore after setup
  [string]$RepoUrl = 'https://github.com/rabbit12345/whisperX',
  [switch]$SkipFFmpeg,                          # FFmpeg shared builds already installed
  [switch]$Yes                                 # non-interactive: accept defaults
)

$ErrorActionPreference = 'Stop'   # native exit codes are checked manually via $LASTEXITCODE
function Info($m){ Write-Host $m -ForegroundColor Cyan }
function Ok($m){ Write-Host $m -ForegroundColor Green }
function Warn($m){ Write-Host $m -ForegroundColor Yellow }

# --- 0. Resolve install location --------------------------------------------
if (-not $InstallRoot) {
  $default = Join-Path $env:USERPROFILE 'Documents\whisperX'
  if ($Yes) { $InstallRoot = $default }
  else {
    $ans = Read-Host "Install location for venvs + data (must be a LOCAL disk) [$default]"
    $InstallRoot = if ([string]::IsNullOrWhiteSpace($ans)) { $default } else { $ans }
  }
}
$InstallRoot = [IO.Path]::GetFullPath($InstallRoot)
if ($InstallRoot -like '\\*') {
  throw "InstallRoot '$InstallRoot' is a UNC/network path. torch-sized installs corrupt over SMB; use a local disk (e.g. under $env:USERPROFILE)."
}
Info "Install root: $InstallRoot"
New-Item -ItemType Directory -Force -Path $InstallRoot, (Join-Path $InstallRoot 'work'),
    (Join-Path $InstallRoot 'downloads') | Out-Null

$WhisperVenv = Join-Path $InstallRoot '.venv-whisperx'
$F5Venv      = Join-Path $InstallRoot '.venv-f5'
$RepoScripts = Join-Path $PSScriptRoot 'scripts'
$Cu128       = 'https://download.pytorch.org/whl/cu128'

# --- 1. Prerequisites --------------------------------------------------------
Info "[1/8] Checking prerequisites..."
function Need($cmd, $hint){ if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { throw "Missing '$cmd'. $hint" } }
Need 'winget' "Install 'App Installer' from the Microsoft Store."
$py311 = $false
if (Get-Command py -ErrorAction SilentlyContinue) {
  & py -3.11 --version *> $null
  $py311 = ($LASTEXITCODE -eq 0)
}
if (-not $py311) {
  Warn "Python 3.11 not found (py -3.11)."
  if ($Yes -or (Read-Host "Install it now via winget? [Y/n]") -notmatch '^n') {
    winget install -e --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
      throw "Python installed, but 'py' isn't on PATH in this shell. Open a NEW terminal and re-run this script."
    }
  } else { throw "Python 3.11 is required (py -3.11)." }
}
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null) -join ', '
  Info "  GPU: $gpu"
} else { Warn "  nvidia-smi not found — GPU acceleration may be unavailable." }

# --- helper: run a venv's python/pip, throwing on failure -------------------
function Venv-Py($venv){ Join-Path $venv 'Scripts\python.exe' }
function Pip($venv, [string[]]$pipArgs){
  & (Venv-Py $venv) -m pip @pipArgs
  if ($LASTEXITCODE -ne 0) { throw "pip $($pipArgs -join ' ') failed (exit $LASTEXITCODE)." }
}

# --- 2. FFmpeg shared builds (7 for whisperx, 8 for f5) ----------------------
if (-not $SkipFFmpeg) {
  Info "[2/8] Installing FFmpeg shared builds (7 + 8)..."
  winget install -e --id BtbN.FFmpeg.LGPL.Shared.7.1 --accept-source-agreements --accept-package-agreements
  winget install -e --id Gyan.FFmpeg.Shared           --accept-source-agreements --accept-package-agreements
} else { Warn "[2/8] -SkipFFmpeg: assuming FFmpeg 7 + 8 shared builds are present." }

# --- 3. WhisperX venv --------------------------------------------------------
Info "[3/8] Creating WhisperX venv -> $WhisperVenv"
& py -3.11 -m venv $WhisperVenv
Pip $WhisperVenv @('install','-U','pip')
Pip $WhisperVenv @('install','whisperx')
Pip $WhisperVenv @('uninstall','-y','torch','torchaudio')
Pip $WhisperVenv @('install','torch==2.8.0','torchaudio==2.8.0','--index-url',$Cu128)
Pip $WhisperVenv @('install','-U','ctranslate2','faster-whisper')   # sm_120 support

# --- 4. F5-TTS venv ----------------------------------------------------------
Info "[4/8] Creating F5-TTS venv -> $F5Venv"
& py -3.11 -m venv $F5Venv
Pip $F5Venv @('install','-U','pip')
Pip $F5Venv @('install','f5-tts')
Pip $F5Venv @('install','--force-reinstall','--no-deps','torch==2.11.0','torchaudio==2.11.0','--index-url',$Cu128)
Pip $F5Venv @('install','--no-deps','torchcodec==0.12.0')

# --- 5. sitecustomize DLL shims (verbatim from the validated setup) ----------
Info "[5/8] Writing sitecustomize.py DLL shims..."
$whisperSite = @'
"""Register the FFmpeg 7 *shared* DLL directory for torchcodec (used by WhisperX).
torchcodec 0.7.0 (torch 2.8.0) supports FFmpeg 4-7 only -> needs avcodec-61.dll.
Python 3.8+ ignores PATH for a DLL's deps, so os.add_dll_directory at startup is
the only thing that works."""
import os, glob
_base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
for _dll in glob.glob(os.path.join(_base, "*", "*", "bin", "avcodec-61.dll")):
    try:
        os.add_dll_directory(os.path.dirname(_dll))
    except OSError:
        pass
    break
'@
$f5Site = @'
"""Register the FFmpeg *shared* DLL directory for torchcodec (used by F5-TTS).
FFmpeg 8 (avcodec-62) shared build, matched to torchcodec 0.12 / torch 2.11.
os.add_dll_directory must run before torchcodec imports -> sitecustomize."""
import os, glob
_base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
for _bin in glob.glob(os.path.join(_base, "Gyan.FFmpeg.Shared*", "*", "bin")):
    if os.path.isdir(_bin):
        try:
            os.add_dll_directory(_bin)
        except OSError:
            pass
        break
'@
Set-Content -Path (Join-Path $WhisperVenv 'Lib\site-packages\sitecustomize.py') -Value $whisperSite -Encoding UTF8
Set-Content -Path (Join-Path $F5Venv      'Lib\site-packages\sitecustomize.py') -Value $f5Site      -Encoding UTF8

# --- 6. Base pinyin vocab for fine-tuning -----------------------------------
Info "[6/8] Placing base pinyin vocab..."
$vocabSrc = Join-Path $F5Venv 'Lib\site-packages\f5_tts\infer\examples\vocab.txt'
$vocabDst = Join-Path $F5Venv 'Lib\data\Emilia_ZH_EN_pinyin\vocab.txt'
if (Test-Path $vocabSrc) {
  New-Item -ItemType Directory -Force -Path (Split-Path $vocabDst) | Out-Null
  Copy-Item $vocabSrc $vocabDst -Force
} else { Warn "  vocab.txt not found at $vocabSrc (Invoke-Finetune will auto-place it on first run)." }

# --- 7. webui/config.json ----------------------------------------------------
Info "[7/8] Writing webui/config.json..."
$cfgPath = Join-Path $PSScriptRoot 'webui\config.json'
$cfg = if (Test-Path $cfgPath) { Get-Content -Raw $cfgPath | ConvertFrom-Json } else { [pscustomobject]@{} }
$fwd = { param($p) $p -replace '\\','/' }
$cfg | Add-Member -Force NoteProperty scripts_dir  (& $fwd $RepoScripts)
$cfg | Add-Member -Force NoteProperty work_dir     (& $fwd (Join-Path $InstallRoot 'work'))
$cfg | Add-Member -Force NoteProperty venv_root    (& $fwd $InstallRoot)
$cfg | Add-Member -Force NoteProperty download_dir (& $fwd (Join-Path $InstallRoot 'downloads'))
if (-not $cfg.PSObject.Properties['download_cmd'])  { $cfg | Add-Member NoteProperty download_cmd '' }
if (-not $cfg.PSObject.Properties['hf_token_env'])  { $cfg | Add-Member NoteProperty hf_token_env 'HF_TOKEN' }
if (-not $cfg.PSObject.Properties['default_epochs']){ $cfg | Add-Member NoteProperty default_epochs 50 }
if (-not $cfg.PSObject.Properties['powershell'])    { $cfg | Add-Member NoteProperty powershell 'pwsh' }
($cfg | ConvertTo-Json -Depth 5) | Set-Content -Path $cfgPath -Encoding UTF8

# HF token (the user's own; stored as a persistent user env var, never printed)
if ($HfToken) {
  setx HF_TOKEN $HfToken | Out-Null
  Ok "  HF_TOKEN saved to your user environment (new shells will see it)."
} else {
  Warn "  No -HfToken given. Diarization needs one: create a read token at"
  Warn "  https://huggingface.co/settings/tokens and accept the pyannote model terms,"
  Warn "  then: setx HF_TOKEN hf_xxx"
}

# --- 8. Verify + optional restore -------------------------------------------
Info "[8/8] Verifying environments..."
& (Venv-Py $WhisperVenv) -c "import torch; from torchcodec.decoders import AudioDecoder; import whisperx; print('whisperx OK', torch.__version__, torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { Warn "  whisperx verify failed — check FFmpeg 7 shared install + driver." }
& (Venv-Py $F5Venv) -c "import torch; from torchcodec.decoders import AudioDecoder; print('f5 OK', torch.__version__, torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { Warn "  f5 verify failed — check FFmpeg 8 shared install + driver." }

if ($RestoreFrom) {
  $restore = Join-Path $PSScriptRoot 'Backup-VoiceCloneData.ps1'
  if (Test-Path $restore) {
    Info "Restoring data from $RestoreFrom ..."
    & $restore -Mode Restore -InstallRoot $InstallRoot -Path $RestoreFrom
  } else { Warn "Backup-VoiceCloneData.ps1 not found; skipping restore." }
}

Ok "`nDone. InstallRoot = $InstallRoot"
Write-Host "Trained models will live in: $F5Venv\Lib\ckpts\<name>\model_last.pt" -ForegroundColor Green
Write-Host "Start the dashboard:  python `"$PSScriptRoot\webui\server.py`" --port 8756" -ForegroundColor Green
