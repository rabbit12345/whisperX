<#
.SYNOPSIS
  One-stop voice-clone pipeline: MKV -> extract -> WhisperX diarize (2 speakers)
  -> pick a speaker's reference clip -> F5-TTS synthesis.

.EXAMPLE
  # 1) See who's talking (diarize only, no synthesis):
  .\Invoke-VoiceClone.ps1 -Mkv "X:\media\clip.mkv" -HfToken $env:HF_TOKEN -ListSpeakers

  # 2) Clone speaker 0 saying some text:
  .\Invoke-VoiceClone.ps1 -Mkv "X:\media\clip.mkv" -HfToken $env:HF_TOKEN `
       -Speaker 0 -GenText "你好，这是克隆的声音。" -OutFile clone.wav

  Re-runs reuse the cached diarization; pass -Force to redo it.
#>
[CmdletBinding(PositionalBinding = $false)]
param(
  [string]$Mkv,              # source recording; OR use -RefAudio for direct synthesis
  [string]$RefAudio,         # synthesize straight from this reference clip (skips diarize/select)
  [ValidateSet('0','1')][string]$Speaker = '0',
  [string]$GenText,
  [string]$HfToken = $env:HF_TOKEN,
  [string]$OutFile = 'clone.wav',
  [string]$WorkDir = "$env:USERPROFILE\Documents\whisperX\work",
  [string]$VenvRoot = "$env:USERPROFILE\Documents\whisperX",
  [string]$Language = 'zh',
  [string]$WhisperModel = 'large-v3',
  [string]$DiarizeModel = 'pyannote/speaker-diarization-3.1',
  [string]$F5Model = 'F5TTS_v1_Base',
  [string]$CkptFile,         # fine-tuned checkpoint (.pt) from Invoke-Finetune; default = base model
  [double]$MinDur = 8.0,     # prefer longer reference for better clone quality
  [double]$MaxDur = 12.0,    # F5-TTS hard-clips the reference at 12s; don't exceed
  # --- F5-TTS quality knobs ---
  [int]$NfeStep = 32,        # denoising steps; 48-64 = cleaner/steadier, slower
  [double]$CfgStrength = 2.0,# guidance; 2.5-3.0 hugs the reference more closely
  [double]$TargetRms = 0.1,  # output loudness normalization
  [double]$Speed = 1.0,      # pacing
  [ValidateSet('vocos','bigvgan')][string]$Vocoder = 'vocos',
  [switch]$RemoveSilence,    # trim long silences in the output
  [switch]$CleanRef,         # gentle highpass + denoise + loudnorm on the reference
  [string]$RefText,          # override reference text (default: WhisperX transcript)
  [switch]$AutoRefText,      # let F5 auto-transcribe the reference instead
  [switch]$Offline,          # force HF offline (no network); all models must already be cached
  [switch]$ListSpeakers,
  [switch]$Force
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'                     # F5-TTS crashes printing zh without this
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
if ($Offline) {
  $env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'; $env:HF_DATASETS_OFFLINE = '1'
}

$WhisperPy  = Join-Path $VenvRoot '.venv-whisperx\Scripts\python.exe'
$WhisperCli = Join-Path $VenvRoot '.venv-whisperx\Scripts\whisperx.exe'
$F5Cli      = Join-Path $VenvRoot '.venv-f5\Scripts\f5-tts_infer-cli.exe'
$SelectPy   = Join-Path $PSScriptRoot 'select_reference.py'

foreach ($p in @($WhisperCli, $F5Cli, $SelectPy)) {
  if (-not (Test-Path $p)) { throw "Missing: $p" }
}
if (-not $GenText -and -not $ListSpeakers) {
  throw "Provide -GenText (or use -ListSpeakers to just inspect speakers)."
}

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

if ($RefAudio) {
  # Direct synthesis: use the supplied reference clip, no diarization/selection.
  if (-not (Test-Path $RefAudio)) { throw "RefAudio not found: $RefAudio" }
  if ($ListSpeakers) { throw "-ListSpeakers needs -Mkv, not -RefAudio." }
  $ref = $RefAudio
  Write-Host "[ref] Using reference clip: $RefAudio" -ForegroundColor DarkGray
} else {
  if (-not $Mkv) { throw "Provide -Mkv (or -RefAudio for direct synthesis)." }
  if (-not (Test-Path $Mkv)) { throw "Input MKV not found: $Mkv" }
  $base  = [IO.Path]::GetFileNameWithoutExtension($Mkv)
  $audio = Join-Path $WorkDir "$base.wav"
  $json  = Join-Path $WorkDir "$base.json"

  # 1. Extract audio (24 kHz mono) -------------------------------------------
  if ($Force -or -not (Test-Path $audio)) {
    Write-Host "[1/4] Extracting audio -> $audio" -ForegroundColor Cyan
    ffmpeg -y -v error -i $Mkv -vn -ac 1 -ar 24000 -c:a pcm_s16le $audio
  } else {
    Write-Host "[1/4] Reusing extracted audio $audio" -ForegroundColor DarkGray
  }

  # 2. Transcribe + diarize (2 speakers) -------------------------------------
  if ($Force -or -not (Test-Path $json)) {
    if (-not $HfToken) { throw "Diarization needs -HfToken (or `$env:HF_TOKEN)." }
    Write-Host "[2/4] WhisperX transcribe + diarize (this is the slow step)..." -ForegroundColor Cyan
    & $WhisperCli $audio --model $WhisperModel --language $Language `
        --diarize --diarize_model $DiarizeModel --hf_token $HfToken `
        --min_speakers 2 --max_speakers 2 `
        --compute_type float16 --output_format json --output_dir $WorkDir
    if ($LASTEXITCODE -ne 0) {
      throw "WhisperX failed (exit $LASTEXITCODE). If it's a 403/GatedRepoError, accept the model terms at https://huggingface.co/$DiarizeModel and https://huggingface.co/pyannote/segmentation-3.0 with your token's account."
    }
    if (-not (Test-Path $json)) { throw "WhisperX ran but produced no $json" }
  } else {
    Write-Host "[2/4] Reusing diarization $json (pass -Force to redo)" -ForegroundColor DarkGray
  }

  # 3. Select the target speaker's reference clip ----------------------------
  $ref = Join-Path $WorkDir ("{0}_speaker{1}_ref.wav" -f $base, $Speaker)
  Write-Host "[3/4] Selecting reference clip for speaker $Speaker..." -ForegroundColor Cyan
  & $WhisperPy $SelectPy --json $json --audio $audio --speaker $Speaker `
      --out $ref --min-dur $MinDur --max-dur $MaxDur
  if ($LASTEXITCODE -ne 0) { throw "Reference selection failed (exit $LASTEXITCODE)." }

  if ($ListSpeakers) {
    Write-Host "`nSpeaker summary printed above. Re-run with -Speaker <0|1> and -GenText to synthesize." -ForegroundColor Yellow
    return
  }
}

# 3b. Optional reference cleanup (gentle highpass + denoise + loudnorm) -------
if ($CleanRef) {
  Write-Host "     Cleaning reference (highpass + denoise + loudnorm)..." -ForegroundColor DarkCyan
  $refClean = "$ref.clean.wav"
  ffmpeg -y -v error -i $ref -af "highpass=f=90,afftdn=nr=10,loudnorm=I=-16:TP=-1.5:LRA=11" `
      -ar 24000 -ac 1 -c:a pcm_s16le $refClean
  if ($LASTEXITCODE -eq 0 -and (Test-Path $refClean)) {
    Move-Item -Force $refClean $ref
  } else {
    Write-Warning "Reference cleanup failed; using the original clip."
  }
}

# 4. Synthesize with F5-TTS --------------------------------------------------
# Accept -OutFile as a bare filename OR a full path.
$OutLeaf = Split-Path -Leaf $OutFile
if ([System.IO.Path]::IsPathRooted($OutFile)) {
  $OutDirEff = Split-Path -Parent $OutFile
} else {
  $OutDirEff = $WorkDir
}
$outPath = Join-Path $OutDirEff $OutLeaf

# Reference text: -AutoRefText (F5 transcribes) > -RefText override > sidecar .txt > auto
if ($AutoRefText) {
  $refText = ''
} elseif ($PSBoundParameters.ContainsKey('RefText')) {
  $refText = $RefText
} elseif (Test-Path "$ref.txt") {
  $refText = (Get-Content -Path "$ref.txt" -Raw -Encoding UTF8).Trim()
} else {
  Write-Host "     No reference transcript found; letting F5 auto-transcribe." -ForegroundColor DarkGray
  $refText = ''
}

$f5Args = @(
  '--model', $F5Model, '--ref_audio', $ref, '--ref_text', $refText,
  '--gen_text', $GenText, '--output_dir', $OutDirEff, '--output_file', $OutLeaf,
  '--nfe_step', $NfeStep, '--cfg_strength', $CfgStrength,
  '--target_rms', $TargetRms, '--speed', $Speed, '--vocoder_name', $Vocoder
)
if ($RemoveSilence) { $f5Args += '--remove_silence' }
if ($CkptFile) { $f5Args += @('--ckpt_file', $CkptFile) }   # use fine-tuned model

Write-Host "[4/4] F5-TTS synthesis (nfe=$NfeStep cfg=$CfgStrength vocoder=$Vocoder) -> $outPath" -ForegroundColor Cyan
& $F5Cli @f5Args
if ($LASTEXITCODE -ne 0) { throw "F5-TTS failed (exit $LASTEXITCODE)." }

if (Test-Path $outPath) {
  Write-Host "`nDONE -> $outPath" -ForegroundColor Green
} else {
  throw "F5-TTS did not produce $outPath"
}
