<#
.SYNOPSIS
  One command: many MKVs -> two fine-tuned voices (female = s0, male = s1).

.DESCRIPTION
  For every supplied recording it diarizes (2 speakers), auto-detects which
  speaker is female vs male by pitch, combines all female audio into one dataset
  and all male audio into another, then fine-tunes both. Reuses the tested
  Invoke-VoiceClone (diarize) and Invoke-Finetune (build + train) scripts.

.EXAMPLE
  $env:HF_TOKEN = "hf_..."
  .\Invoke-VoiceClones.ps1 -Mkvs @(
      "C:\...\ep446.mkv", "C:\...\ep448.mkv", "C:\...\ep450.mkv") -Epochs 100
#>
[CmdletBinding(PositionalBinding = $false)]
param(
  [Parameter(Mandatory)][string[]]$Mkvs,       # all recordings to use
  [string]$HfToken = $env:HF_TOKEN,
  [string]$WorkDir = "$env:USERPROFILE\Documents\whisperX\work",
  [string]$VenvRoot = "$env:USERPROFILE\Documents\whisperX",
  [string]$FemaleDataset = 's0',
  [string]$MaleDataset = 's1',
  [int]$Epochs = 100,
  [ValidateSet('both', 'female', 'male')][string]$Train = 'both',
  [switch]$NoTrain,                            # stop after diarize + gender preview (no training)
  [switch]$Offline
)

$ErrorActionPreference = 'Stop'
if (-not $HfToken) {
  $tokenFile = Join-Path (Split-Path $PSScriptRoot -Parent) 'HF_Token.txt'
  if (Test-Path $tokenFile) { $HfToken = (Get-Content $tokenFile -Raw).Trim() }
}
$env:PYTHONUTF8 = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
if ($Offline) {
  $env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'; $env:HF_DATASETS_OFFLINE = '1'
}

$F5Py       = Join-Path $VenvRoot '.venv-f5\Scripts\python.exe'
$ClassifyPy = Join-Path $PSScriptRoot 'classify_gender.py'
$VoiceClone = Join-Path $PSScriptRoot 'Invoke-VoiceClone.ps1'
$Finetune   = Join-Path $PSScriptRoot 'Invoke-Finetune.ps1'
foreach ($p in @($F5Py, $ClassifyPy, $VoiceClone, $Finetune)) {
  if (-not (Test-Path $p)) { throw "Missing: $p" }
}

function Confirm-Diarized([string]$mkvPath) {
  $b = [IO.Path]::GetFileNameWithoutExtension($mkvPath)
  $j = Join-Path $WorkDir "$b.json"
  if (Test-Path $j) { Write-Host "   diarization cached: $b" -ForegroundColor DarkGray; return }
  if (-not (Test-Path $mkvPath)) { throw "MKV not found: $mkvPath" }
  if (-not $HfToken) { throw "Need -HfToken (or `$env:HF_TOKEN) to diarize '$b'." }
  Write-Host "   diarizing $b (GPU)..." -ForegroundColor Cyan
  # Hashtable splat (NOT array) so named params bind correctly to the child script.
  $vc = @{ Mkv = $mkvPath; HfToken = $HfToken; WorkDir = $WorkDir; VenvRoot = $VenvRoot; ListSpeakers = $true }
  if ($Offline) { $vc.Offline = $true }
  & $VoiceClone @vc
  if (-not (Test-Path $j)) { throw "Diarization did not produce $j for '$b'." }
}

# --- Stage 1: diarize + gender-classify every recording ----------------------
Write-Host "=== Stage 1: diarize + gender-classify $($Mkvs.Count) recording(s) ===" -ForegroundColor Cyan
$femaleSources = @()
$maleSources = @()
foreach ($mkv in $Mkvs) {
  $b = [IO.Path]::GetFileNameWithoutExtension($mkv)
  Confirm-Diarized $mkv
  $audio = Join-Path $WorkDir "$b.wav"
  $json  = Join-Path $WorkDir "$b.json"
  $out = (& $F5Py $ClassifyPy --audio $audio --json $json)
  if ($LASTEXITCODE -ne 0) { throw "Gender classification failed for '$b'." }
  $idx = (($out | Select-Object -Last 1).ToString().Trim() -split '\s+')
  $femIdx, $maleIdx = $idx[0], $idx[1]
  if ($femIdx -eq '-1' -and $maleIdx -eq '-1') {
    Write-Host ("   {0} -> SKIPPED (low-confidence gender split; excluded from training)" -f $b) -ForegroundColor Yellow
  } else {
    Write-Host ("   {0} -> female=SPEAKER_{1:d2} male=SPEAKER_{2:d2}" -f $b, [int]$femIdx, [int]$maleIdx) -ForegroundColor DarkCyan
  }
  if ($femIdx -ne '-1') { $femaleSources += "$mkv|$femIdx" }
  if ($maleIdx -ne '-1') { $maleSources += "$mkv|$maleIdx" }
}

Write-Host ("Assignment: Female='{0}' <- {1} recording(s); Male='{2}' <- {3} recording(s)." -f `
    $FemaleDataset, $femaleSources.Count, $MaleDataset, $maleSources.Count) -ForegroundColor Yellow
if ($NoTrain) {
  Write-Host "-NoTrain: stopping after gender preview. Re-run without it to fine-tune." -ForegroundColor Yellow
  return
}

# --- Stage 2: fine-tune the two voices ---------------------------------------
Write-Host "=== Stage 2: fine-tune voices ===" -ForegroundColor Cyan
function Invoke-VoiceFinetune([string]$name, [string[]]$sources, [string]$gender) {
  if (-not $sources) { throw "No $gender speaker found across the supplied recordings." }
  Write-Host "--- $gender -> $name  ($($sources.Count) recording(s)) ---" -ForegroundColor Green
  $ft = @{
    DatasetName = $name; Sources = $sources; Epochs = $Epochs
    WorkDir = $WorkDir; VenvRoot = $VenvRoot; HfToken = $HfToken
    Force = $true            # assemble the dataset from exactly the supplied MKVs each run
  }
  if ($Offline) { $ft.Offline = $true }
  & $Finetune @ft
  if ($LASTEXITCODE -ne 0) { throw "$gender fine-tune failed." }
}

if ($Train -in 'both', 'female') { Invoke-VoiceFinetune $FemaleDataset $femaleSources 'Female' }
if ($Train -in 'both', 'male')   { Invoke-VoiceFinetune $MaleDataset   $maleSources   'Male' }

Write-Host "`nAll done. Female='$FemaleDataset', Male='$MaleDataset'." -ForegroundColor Green
Write-Host "Infer, e.g.:  .\Invoke-VoiceClone.ps1 -Mkv `"$($Mkvs[0])`" -Speaker 0 -GenText `"...`" -CkptFile `"$VenvRoot\.venv-f5\Lib\ckpts\$FemaleDataset\model_last.pt`"" -ForegroundColor Green
