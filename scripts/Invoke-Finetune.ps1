<#
.SYNOPSIS
  Streamlined F5-TTS fine-tuning for one diarized speaker.
  MKV/cached diarization -> per-speaker dataset -> prepare -> fine-tune.

.DESCRIPTION
  Reuses the cached WhisperX diarization from Invoke-VoiceClone (same WorkDir).
  Run that first (at least once, so <name>.json exists), then this. Produces a
  fine-tuned checkpoint under <venv>\Lib\ckpts\<DatasetName>\ that you pass back
  to Invoke-VoiceClone via -CkptFile for inference.

.EXAMPLE
  .\Invoke-Finetune.ps1 -Mkv "C:\...\clip.mkv" -Speaker 0 -DatasetName s0 -Epochs 100
#>
[CmdletBinding(PositionalBinding = $false)]   # all args must be named; a stray token errors clearly
param(
  [string]$Mkv,                      # single source; or use -Sources for multiple recordings
  [ValidateSet('0','1')][string]$Speaker = '0',
  [string[]]$Sources,               # each "mkvpath|speaker", e.g. @("C:\ep1.mkv|0","C:\ep2.mkv|1")
  [string[]]$ExtraMkvs,             # with -AutoMatch: extra recordings; target label auto-detected
  [switch]$AutoMatch,               # use -Mkv/-Speaker as reference, auto-find that voice in -ExtraMkvs
  [string]$HfToken = $env:HF_TOKEN, # only needed to auto-diarize recordings not yet cached
  [string]$DatasetName = 's0',
  [string]$WorkDir = "$env:USERPROFILE\Documents\whisperX\work",
  [string]$VenvRoot = "$env:USERPROFILE\Documents\whisperX",
  [string]$ExpName = 'F5TTS_v1_Base',
  # --- training hyperparameters (defaults tuned for ~15 min of data) ---
  [double]$LearningRate = 1e-5,
  [int]$BatchSizePerGpu = 2400,      # frames; smaller = less padding waste on variable-length clips (much faster here)
  [ValidateSet('bf16','fp16','no')][string]$Precision = 'bf16',  # 5090: bf16 ~2x faster than fp32
  [int]$Epochs = 100,
  [int]$NumWarmupUpdates = 100,
  [int]$SavePerUpdates = 200,        # checkpoint cadence; listen to samples, stop when good
  [int]$LastPerUpdates = 100,
  [int]$KeepLastN = 3,
  [ValidateSet('none','tensorboard','wandb')][string]$Logger = 'none',
  [switch]$Offline,                  # force HF offline; base model must be cached
  [switch]$Force,                    # rebuild dataset even if it exists
  [switch]$NoPurityGate,             # skip the embedding purity gate (keeps overlap-contaminated clips)
  [string]$HistoryFile               # training-history JSON (default: <WorkDir>\training_history.json)
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:ACCELERATE_MIXED_PRECISION = $Precision   # accelerate reads this when trainer passes no explicit value
if ($Offline) {
  $env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'; $env:HF_DATASETS_OFFLINE = '1'
}

$F5Py     = Join-Path $VenvRoot '.venv-f5\Scripts\python.exe'
$F5Ft     = Join-Path $VenvRoot '.venv-f5\Scripts\f5-tts_finetune-cli.exe'
$WhisperPy = Join-Path $VenvRoot '.venv-whisperx\Scripts\python.exe'
$BuildPy  = Join-Path $PSScriptRoot 'build_finetune_dataset.py'
$MatchPy  = Join-Path $PSScriptRoot 'match_speaker.py'
$FilterPy = Join-Path $PSScriptRoot 'filter_dataset.py'
foreach ($p in @($F5Py, $F5Ft, $BuildPy)) { if (-not (Test-Path $p)) { throw "Missing: $p" } }

if (-not $HistoryFile) { $HistoryFile = Join-Path $WorkDir 'training_history.json' }

# --- training-history helpers ------------------------------------------------
function Get-MkvMeta([string]$path) {
  # File size, modified date, and container duration (ffprobe) for one source MKV.
  # The MKV may have been deleted after diarization (cached wav/json suffice for
  # training), so metadata is best-effort.
  if (-not (Test-Path -LiteralPath $path)) {
    return [ordered]@{
      mkv          = $path
      file         = [IO.Path]::GetFileName($path)
      size_bytes   = $null
      size_mb      = $null
      modified     = $null
      duration_sec = $null
    }
  }
  $fi = Get-Item -LiteralPath $path
  $dur = $null
  try {
    $probe = & ffprobe -v error -show_entries format=duration -of csv=p=0 -- $path 2>$null
    if ($probe) { $dur = [math]::Round([double]($probe | Select-Object -Last 1), 1) }
  } catch { }
  [ordered]@{
    mkv          = $path
    file         = $fi.Name
    size_bytes   = $fi.Length
    size_mb      = [math]::Round($fi.Length / 1MB, 1)
    modified     = $fi.LastWriteTime.ToString('o')
    duration_sec = $dur
  }
}

function Add-HistoryRecord([string]$file, $record) {
  # Append one record to the JSON array file (created if missing), tolerant of a
  # missing/corrupt/single-object existing file.
  $list = @()
  if (Test-Path $file) {
    try { $list = @(Get-Content -Raw -LiteralPath $file | ConvertFrom-Json) } catch { $list = @() }
  }
  $list += [pscustomobject]$record
  New-Item -ItemType Directory -Force -Path (Split-Path $file) | Out-Null
  # ConvertTo-Json collapses a 1-element array into a bare object; force array form
  # so consumers always get a JSON list.
  $json = @($list) | ConvertTo-Json -Depth 8
  if (@($list).Count -eq 1) { $json = "[$json]" }
  $json | Set-Content -LiteralPath $file -Encoding UTF8
}

# Per-source metadata gathered during the build stage (empty when dataset reused).
$sourceRecords = @()

# Auto-diarize a recording if its cached JSON is missing, by calling the tested
# Invoke-VoiceClone -ListSpeakers path (extract audio + WhisperX diarize + cache).
$VoiceClone = Join-Path $PSScriptRoot 'Invoke-VoiceClone.ps1'
function Confirm-Diarized([string]$mkvPath) {
  $b = [IO.Path]::GetFileNameWithoutExtension($mkvPath)
  $j = Join-Path $WorkDir "$b.json"
  if (Test-Path $j) { Write-Host "   diarization cached: $b" -ForegroundColor DarkGray; return }
  if (-not (Test-Path $mkvPath)) { throw "MKV not found: $mkvPath" }
  if (-not $HfToken) { throw "Need -HfToken (or `$env:HF_TOKEN) to diarize '$b' (not yet cached)." }
  Write-Host "   diarizing $b (first time; uses GPU)..." -ForegroundColor Cyan
  # Hashtable splat (NOT array) so named params bind correctly to the child script.
  $vc = @{ Mkv = $mkvPath; HfToken = $HfToken; WorkDir = $WorkDir; VenvRoot = $VenvRoot; ListSpeakers = $true }
  if ($Offline) { $vc.Offline = $true }
  & $VoiceClone @vc
  if (-not (Test-Path $j)) { throw "Diarization did not produce $j for '$b'." }
}

# Resolve F5-TTS data dir and ensure the base pinyin vocab is present.
$dataDir = (& $F5Py -c "from importlib.resources import files; import os; print(os.path.abspath(str(files('f5_tts').joinpath('../../data'))))").Trim()
$emiliaVocab = Join-Path $dataDir 'Emilia_ZH_EN_pinyin\vocab.txt'
if (-not (Test-Path $emiliaVocab)) {
  Write-Host "[setup] Placing base pinyin vocab..." -ForegroundColor Cyan
  New-Item -ItemType Directory -Force -Path (Split-Path $emiliaVocab) | Out-Null
  Copy-Item (Join-Path $VenvRoot '.venv-f5\Lib\site-packages\f5_tts\infer\examples\vocab.txt') $emiliaVocab
}

# 1. Build the combined dataset (word-chunked clips + metadata.csv) -----------
# Only diarize / match / build when the dataset actually needs building; a
# prepared dataset is re-trained without touching the source MKVs at all.
$stage = Join-Path $WorkDir "finetune\$DatasetName"
$srcList = $null
if ($Force -or -not (Test-Path (Join-Path $stage 'metadata.csv'))) {

  # Collect every recording involved, then ensure each is diarized.
  if ($AutoMatch) {
    if (-not $Mkv)       { throw "-AutoMatch needs -Mkv + -Speaker as the reference recording." }
    if (-not $ExtraMkvs) { throw "-AutoMatch needs -ExtraMkvs (the other recordings to match)." }
    if (-not (Test-Path $WhisperPy)) { throw "Missing: $WhisperPy" }
    $allMkvs = @($Mkv) + $ExtraMkvs
  } elseif ($Sources) {
    $allMkvs = $Sources | ForEach-Object {
      $parts = $_ -split '\|', 2
      if ($parts.Count -ne 2) { throw "Bad -Sources entry '$_'. Use 'C:\path\ep.mkv|0'." }
      $parts[0].Trim()
    }
  } elseif ($Mkv) {
    $allMkvs = @($Mkv)
  } else {
    throw "Provide -Mkv (+ -Speaker), -Sources, or -Mkv + -ExtraMkvs -AutoMatch."
  }
  Write-Host "[0/3] Ensuring diarization for $(@($allMkvs).Count) recording(s)..." -ForegroundColor Cyan
  foreach ($m in $allMkvs) { Confirm-Diarized $m }

  # Resolve the {Mkv, Speaker} list (JSONs now present).
  if ($AutoMatch) {
    $refB     = [IO.Path]::GetFileNameWithoutExtension($Mkv)
    $refAudio = Join-Path $WorkDir "$refB.wav"
    $refJson  = Join-Path $WorkDir "$refB.json"
    $srcList  = @([pscustomobject]@{ Mkv = $Mkv; Speaker = $Speaker })
    foreach ($ex in $ExtraMkvs) {
      $exB     = [IO.Path]::GetFileNameWithoutExtension($ex)
      $exAudio = Join-Path $WorkDir "$exB.wav"
      $exJson  = Join-Path $WorkDir "$exB.json"
      $lab = (& $WhisperPy $MatchPy --ref-audio $refAudio --ref-json $refJson --ref-speaker $Speaker `
                --audio $exAudio --json $exJson)
      if ($LASTEXITCODE -ne 0) { throw "Speaker match failed for '$ex'." }
      $lab = ($lab | Select-Object -Last 1).ToString().Trim()
      Write-Host "   auto-matched $exB -> speaker $lab" -ForegroundColor DarkCyan
      $srcList += [pscustomobject]@{ Mkv = $ex; Speaker = $lab }
    }
  } elseif ($Sources) {
    $srcList = foreach ($s in $Sources) {
      $parts = $s -split '\|', 2
      [pscustomobject]@{ Mkv = $parts[0].Trim(); Speaker = $parts[1].Trim() }
    }
  } else {
    $srcList = @([pscustomobject]@{ Mkv = $Mkv; Speaker = $Speaker })
  }

  Write-Host "[1/3] Building dataset from $($srcList.Count) source(s)..." -ForegroundColor Cyan
  if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
  $i = 0
  foreach ($src in $srcList) {
    $b = [IO.Path]::GetFileNameWithoutExtension($src.Mkv)
    $a = Join-Path $WorkDir "$b.wav"
    $j = Join-Path $WorkDir "$b.json"
    if (-not (Test-Path $a)) { throw "Missing extracted audio $a (run Invoke-VoiceClone -ListSpeakers)." }
    Write-Host "   + $b  (speaker $($src.Speaker))" -ForegroundColor DarkCyan
    # Clips this source will add start past the highest existing index (matches the
    # build script's append numbering, which skips gaps left by the purity gate).
    $wavDir = Join-Path $stage 'wavs'
    $startIdx = 0
    if (Test-Path $wavDir) {
      $maxIdx = Get-ChildItem $wavDir -Filter 'seg_*.wav' |
        ForEach-Object { [int]$_.BaseName.Substring(4) } | Measure-Object -Maximum
      if ($maxIdx.Count -gt 0) { $startIdx = [int]$maxIdx.Maximum + 1 }
    }
    $buildArgs = @($BuildPy, '--json', $j, '--audio', $a, '--speaker', $src.Speaker, '--out-dir', $stage)
    if ($i -gt 0) { $buildArgs += '--append' }
    # Capture output (build script reports "kept N clips (M.M min)" on stderr) while still showing it.
    $buildOut = & $F5Py @buildArgs 2>&1
    if ($LASTEXITCODE -ne 0) { $buildOut | Out-Host; throw "Dataset build failed for $b (exit $LASTEXITCODE)." }
    $buildOut | Out-Host
    if (-not $NoPurityGate) {
      # Embedding purity gate: drop this source's clips that contain any window of
      # the other speaker's voice (overlapped speech the word labels can't see).
      if (-not (Test-Path $WhisperPy)) { throw "Missing: $WhisperPy (needed for purity gate)" }
      & $WhisperPy $FilterPy --json $j --audio $a --speaker $src.Speaker `
          --wav-dir $wavDir --meta (Join-Path $stage 'metadata.csv') --start-index $startIdx
      if ($LASTEXITCODE -ne 0) { throw "Purity gate failed for $b (exit $LASTEXITCODE)." }
    }
    $km = [regex]::Match(($buildOut -join "`n"), 'kept (\d+) clips \(([\d.]+) min\)')
    $meta = Get-MkvMeta $src.Mkv
    $meta.speaker      = $src.Speaker
    $meta.clips_kept   = if ($km.Success) { [int]$km.Groups[1].Value } else { $null }
    $meta.clip_minutes = if ($km.Success) { [double]$km.Groups[2].Value } else { $null }
    $sourceRecords += [pscustomobject]$meta
    $i++
  }
} else {
  Write-Host "[1/3] Reusing dataset $stage (pass -Force to rebuild; MKVs not needed)" -ForegroundColor DarkGray
}

# 2. Prepare arrow dataset (omit --pretrain => is_finetune, reuse base vocab) -
$prepared = Join-Path $dataDir "${DatasetName}_pinyin"
if ($Force -or -not (Test-Path (Join-Path $prepared 'raw.arrow'))) {
  Write-Host "[2/3] Preparing arrow dataset -> $prepared" -ForegroundColor Cyan
  if (Test-Path $prepared) { Remove-Item -Recurse -Force $prepared }
  & $F5Py -m f5_tts.train.datasets.prepare_csv_wavs (Join-Path $stage 'metadata.csv') $prepared
  if ($LASTEXITCODE -ne 0) { throw "prepare_csv_wavs failed (exit $LASTEXITCODE)." }
} else {
  Write-Host "[2/3] Reusing prepared dataset $prepared" -ForegroundColor DarkGray
}

# 3. Fine-tune ---------------------------------------------------------------
Write-Host "[3/3] Fine-tuning $ExpName on '$DatasetName' (Ctrl-C to stop; checkpoints saved every $SavePerUpdates updates)..." -ForegroundColor Cyan
$ftArgs = @(
  '--exp_name', $ExpName, '--dataset_name', $DatasetName, '--finetune', '--tokenizer', 'pinyin',
  '--learning_rate', $LearningRate, '--batch_size_per_gpu', $BatchSizePerGpu, '--batch_size_type', 'frame',
  '--epochs', $Epochs, '--num_warmup_updates', $NumWarmupUpdates,
  '--save_per_updates', $SavePerUpdates, '--last_per_updates', $LastPerUpdates,
  '--keep_last_n_checkpoints', $KeepLastN, '--log_samples'
)
if ($Logger -ne 'none') { $ftArgs += @('--logger', $Logger) }   # omit => no logger (default None)
if ($Offline) {
  # Point --pretrain at the locally cached base checkpoint so it never resolves hf://
  $pre = Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub\models--SWivid--F5-TTS\snapshots" `
           -Recurse -Filter 'model_1250000.safetensors' -ErrorAction SilentlyContinue |
         Select-Object -First 1 -ExpandProperty FullName
  if (-not $pre) { throw "Offline: base checkpoint not cached. Run once online first." }
  $ftArgs += @('--pretrain', $pre)
}
$trainSw = [System.Diagnostics.Stopwatch]::StartNew()
& $F5Ft @ftArgs
$trainSw.Stop()
if ($LASTEXITCODE -ne 0) { throw "Fine-tuning failed (exit $LASTEXITCODE)." }

$ckptDir = (& $F5Py -c "from importlib.resources import files; import os; print(os.path.abspath(str(files('f5_tts').joinpath('../../ckpts/$DatasetName'))))").Trim()

# --- Record this run in the training history --------------------------------
$totalClips = $null
$metaCsv = Join-Path $stage 'metadata.csv'
if (Test-Path $metaCsv) { $totalClips = @(Get-Content -LiteralPath $metaCsv).Count - 1 }   # minus header
$totalMinutes = if ($sourceRecords -and ($sourceRecords.clip_minutes -notcontains $null)) {
  [math]::Round((($sourceRecords | Measure-Object clip_minutes -Sum).Sum), 1)
} else { $null }
$record = [ordered]@{
  timestamp          = (Get-Date).ToString('o')
  dataset            = $DatasetName
  exp_name           = $ExpName
  epochs             = $Epochs
  batch_size_per_gpu = $BatchSizePerGpu
  precision          = $Precision
  learning_rate      = $LearningRate
  offline            = [bool]$Offline
  dataset_rebuilt    = [bool]($srcList)             # $srcList set only when we (re)built this run
  training_seconds   = [math]::Round($trainSw.Elapsed.TotalSeconds, 1)
  training_hms       = $trainSw.Elapsed.ToString('hh\:mm\:ss')
  total_clips        = $totalClips
  total_minutes      = $totalMinutes
  checkpoint_dir     = $ckptDir
  sources            = @($sourceRecords)
}
try {
  Add-HistoryRecord $HistoryFile $record
  Write-Host "History updated -> $HistoryFile" -ForegroundColor DarkGray
} catch {
  Write-Warning "Could not write training history ($HistoryFile): $($_.Exception.Message)"
}
$hintMkv = if ($srcList) { $srcList[0].Mkv } elseif ($Mkv) { $Mkv } else { '<source.mkv>' }
$hintSpk = if ($srcList) { $srcList[0].Speaker } else { $Speaker }
Write-Host "`nDONE. Checkpoints in: $ckptDir" -ForegroundColor Green
Write-Host "Use one for inference:" -ForegroundColor Green
Write-Host "  .\Invoke-VoiceClone.ps1 -Mkv `"$hintMkv`" -Speaker $hintSpk -GenText `"...`" -CkptFile `"$ckptDir\model_last.pt`"" -ForegroundColor Green
