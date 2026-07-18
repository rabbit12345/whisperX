<#
.SYNOPSIS
  Wipe fine-tuned voice training state so the next run starts from the base model.

.DESCRIPTION
  For each dataset (default s0, s1) deletes:
    * checkpoints    <f5 ckpts>\<name>          (else the trainer resumes instead of restarting)
    * prepared data  <f5 data>\<name>_pinyin
    * staged dataset <WorkDir>\finetune\<name>
  and clears the training-history JSON (a .bak copy is kept). After this,
  Invoke-VoiceClones / Invoke-Finetune fine-tune from the pretrained base
  checkpoint instead of continuing a previous run.

.EXAMPLE
  .\Reset-Training.ps1 -Yes
#>
[CmdletBinding(PositionalBinding = $false)]
param(
  [string[]]$Datasets = @('s0', 's1'),
  [string]$WorkDir = "$env:USERPROFILE\Documents\whisperX\work",
  [string]$VenvRoot = "$env:USERPROFILE\Documents\whisperX",
  [string]$HistoryFile,
  [switch]$Yes                       # skip the confirmation prompt (required from the webui)
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'

$F5Py = Join-Path $VenvRoot '.venv-f5\Scripts\python.exe'
if (-not (Test-Path $F5Py)) { throw "Missing: $F5Py" }
if (-not $HistoryFile) { $HistoryFile = Join-Path $WorkDir 'training_history.json' }

# Resolve the F5-TTS data + ckpts roots exactly as Invoke-Finetune does.
$dataDir  = (& $F5Py -c "from importlib.resources import files; import os; print(os.path.abspath(str(files('f5_tts').joinpath('../../data'))))").Trim()
$ckptRoot = (& $F5Py -c "from importlib.resources import files; import os; print(os.path.abspath(str(files('f5_tts').joinpath('../../ckpts'))))").Trim()

$targets = @()
foreach ($d in $Datasets) {
  $targets += Join-Path $ckptRoot $d
  $targets += Join-Path $dataDir "${d}_pinyin"
  $targets += Join-Path $WorkDir "finetune\$d"
}

Write-Host "Reset targets:" -ForegroundColor Yellow
foreach ($t in $targets) {
  $mark = if (Test-Path $t) { '[exists]' } else { '[absent]' }
  Write-Host "  $mark $t"
}
Write-Host "Clear history: $HistoryFile" -ForegroundColor Yellow

if (-not $Yes) {
  $ans = Read-Host "Delete the above and start training from scratch? (y/N)"
  if ($ans -ne 'y') { Write-Host 'Aborted.' -ForegroundColor DarkGray; return }
}

foreach ($t in $targets) {
  if (Test-Path $t) {
    Remove-Item -Recurse -Force $t
    Write-Host "deleted $t" -ForegroundColor DarkGray
  }
}

if (Test-Path $HistoryFile) {
  Copy-Item $HistoryFile "$HistoryFile.bak" -Force
  '[]' | Set-Content -LiteralPath $HistoryFile -Encoding UTF8
  Write-Host "cleared history (backup: $HistoryFile.bak)" -ForegroundColor DarkGray
}

Write-Host "Reset complete. The next fine-tune starts from the base model." -ForegroundColor Green
