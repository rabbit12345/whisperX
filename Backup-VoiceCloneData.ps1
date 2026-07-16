<#
.SYNOPSIS
  Back up (or restore) the voice-clone workflow's DATA — the parts that are
  expensive or impossible to regenerate — so a fresh machine can be brought up
  with Install-VoiceCloneWorkflow.ps1 + a restore.

.DESCRIPTION
  Captured by default:
    * trained checkpoints  .venv-f5\Lib\ckpts\<name>\model_last.pt (+ pretrained, samples)
    * prepared datasets    .venv-f5\Lib\data\<name>_pinyin\  (raw.arrow, duration.json, vocab.txt)
    * work metadata + diarization cache  work\*.json
        (training_history.json, downloads_index.json, playlist_analysis.json, <ep>.json)
  NOT captured (recreate via the installer / re-download):
    * the venvs themselves, and the HuggingFace model cache (unless -IncludeModels)
    * extracted work\*.wav (unless -IncludeAudio) and source MKVs (unless -IncludeDownloads)

.EXAMPLE
  # Back up to an external drive:
  .\Backup-VoiceCloneData.ps1 -Mode Backup -Path E:\backups\voice-2026-07-16

.EXAMPLE
  # On a fresh machine, after running the installer:
  .\Backup-VoiceCloneData.ps1 -Mode Restore -Path E:\backups\voice-2026-07-16 -InstallRoot C:\voice
#>
[CmdletBinding()]
param(
  [ValidateSet('Backup','Restore')][string]$Mode = 'Backup',
  [Parameter(Mandatory)][string]$Path,                    # backup folder (dest for Backup, src for Restore)
  [string]$InstallRoot = (Join-Path $env:USERPROFILE 'Documents\whisperX'),
  [switch]$AllCheckpoints,     # include every model_*.pt (default: only model_last.pt)
  [switch]$IncludeAudio,       # include work\*.wav (large; regenerable from MKV)
  [switch]$IncludeDownloads,   # include downloads\*.mkv (large)
  [switch]$IncludeModels       # include HF model cache ~/.cache/huggingface/hub (large; re-downloadable)
)

$ErrorActionPreference = 'Stop'
function Info($m){ Write-Host $m -ForegroundColor Cyan }
function Ok($m){ Write-Host $m -ForegroundColor Green }
function Warn($m){ Write-Host $m -ForegroundColor Yellow }

# robocopy wrapper: exit codes 0-7 are success (8+ = real error).
function Robo($src, $dst, [string[]]$files, [switch]$Recurse){
  if (-not (Test-Path $src)) { Warn "  skip (missing): $src"; return }
  $args = @($src, $dst) + $files + @('/NFL','/NDL','/NJH','/NJS','/NP','/R:1','/W:1')
  if ($Recurse) { $args += '/E' }
  robocopy @args | Out-Null
  if ($LASTEXITCODE -ge 8) { throw "robocopy '$src' -> '$dst' failed (code $LASTEXITCODE)." }
}

$F5     = Join-Path $InstallRoot '.venv-f5'
$Ckpts  = Join-Path $F5 'Lib\ckpts'
$Data   = Join-Path $F5 'Lib\data'
$Work   = Join-Path $InstallRoot 'work'
$Dl     = Join-Path $InstallRoot 'downloads'
$HfCache = Join-Path $env:USERPROFILE '.cache\huggingface\hub'

if ($Mode -eq 'Backup') {
  Info "Backing up $InstallRoot  ->  $Path"
  New-Item -ItemType Directory -Force -Path $Path | Out-Null

  # 1. Trained checkpoints (per dataset)
  if (Test-Path $Ckpts) {
    foreach ($d in Get-ChildItem -Directory $Ckpts) {
      $dst = Join-Path $Path "ckpts\$($d.Name)"
      if ($AllCheckpoints) { Robo $d.FullName $dst @() -Recurse }
      else {
        Robo $d.FullName $dst @('model_last.pt','pretrained_*.safetensors')
        Robo (Join-Path $d.FullName 'samples') (Join-Path $dst 'samples') @() -Recurse
      }
      Info "  ckpts\$($d.Name)"
    }
  } else { Warn "  no checkpoints found at $Ckpts" }

  # 2. Prepared datasets (skip the base Emilia vocab dir — reinstalled)
  if (Test-Path $Data) {
    foreach ($d in Get-ChildItem -Directory $Data | Where-Object Name -like '*_pinyin' |
                   Where-Object Name -ne 'Emilia_ZH_EN_pinyin') {
      Robo $d.FullName (Join-Path $Path "data\$($d.Name)") @() -Recurse
      Info "  data\$($d.Name)"
    }
  }

  # 3. work: metadata + diarization JSON always; wavs only if asked
  $workFiles = @('*.json')
  if ($IncludeAudio) { $workFiles += '*.wav'; $workFiles += '*.txt' }
  Robo $Work (Join-Path $Path 'work') $workFiles
  Info "  work (json$([bool]$IncludeAudio ? ' + wav' : ''))"

  if ($IncludeDownloads) { Robo $Dl (Join-Path $Path 'downloads') @('*.mkv'); Info "  downloads\*.mkv" }
  if ($IncludeModels)    { Robo $HfCache (Join-Path $Path 'hf-cache') @() -Recurse; Info "  hf-cache" }

  $manifest = [ordered]@{
    created         = (Get-Date).ToString('o')
    install_root    = $InstallRoot
    all_checkpoints = [bool]$AllCheckpoints
    include_audio   = [bool]$IncludeAudio
    include_downloads = [bool]$IncludeDownloads
    include_models  = [bool]$IncludeModels
  }
  ($manifest | ConvertTo-Json) | Set-Content (Join-Path $Path 'backup-manifest.json') -Encoding UTF8
  $size = (Get-ChildItem -Recurse $Path | Measure-Object Length -Sum).Sum / 1GB
  Ok ("Backup complete: {0:N1} GB at {1}" -f $size, $Path)
}
else {
  # --- Restore -----------------------------------------------------------------
  if (-not (Test-Path $Path)) { throw "Backup not found: $Path" }
  if (-not (Test-Path $F5))   { throw "$F5 missing — run Install-VoiceCloneWorkflow.ps1 first, then restore." }
  Info "Restoring $Path  ->  $InstallRoot"

  if (Test-Path (Join-Path $Path 'ckpts')) {
    foreach ($d in Get-ChildItem -Directory (Join-Path $Path 'ckpts')) {
      Robo $d.FullName (Join-Path $Ckpts $d.Name) @() -Recurse; Info "  ckpts\$($d.Name)"
    }
  }
  if (Test-Path (Join-Path $Path 'data')) {
    foreach ($d in Get-ChildItem -Directory (Join-Path $Path 'data')) {
      Robo $d.FullName (Join-Path $Data $d.Name) @() -Recurse; Info "  data\$($d.Name)"
    }
  }
  if (Test-Path (Join-Path $Path 'work'))      { Robo (Join-Path $Path 'work') $Work @() -Recurse; Info "  work" }
  if (Test-Path (Join-Path $Path 'downloads')) { Robo (Join-Path $Path 'downloads') $Dl @() -Recurse; Info "  downloads" }
  if (Test-Path (Join-Path $Path 'hf-cache'))  { Robo (Join-Path $Path 'hf-cache') $HfCache @() -Recurse; Info "  hf-cache" }
  Ok "Restore complete into $InstallRoot"
}
exit 0   # robocopy leaves a nonzero-but-successful $LASTEXITCODE; report clean success
