<#
Manage-Dashboard.ps1 - start/restart/stop the voice-clone dashboard (webui/server.py).

Default behavior: if the server is not running, start it; if it is running, restart it.
Use -Stop to shut it down.

Usage:
  .\Manage-Dashboard.ps1            # start or restart
  .\Manage-Dashboard.ps1 -Stop     # stop
  .\Manage-Dashboard.ps1 -Port 8756
#>
param(
    [int]$Port = 8756,
    [int]$WorkerPort = 8757,
    [switch]$Stop,
    [switch]$NoWorker    # skip the GPU TTS worker (frees VRAM; TTS falls back to the slow path)
)

$ErrorActionPreference = 'Stop'
$serverScript = Join-Path $PSScriptRoot 'webui\server.py'
$workerScript = Join-Path $PSScriptRoot 'webui\tts_worker.py'
$logFile      = Join-Path $PSScriptRoot 'webui\server.log'
$workerLog    = Join-Path $PSScriptRoot 'webui\tts_worker.log'
$config       = Get-Content (Join-Path $PSScriptRoot 'webui\config.json') -Raw | ConvertFrom-Json
$f5Python     = Join-Path $config.venv_root '.venv-f5\Scripts\python.exe'

function Get-PortProcess([int]$p) {
    # Find the process listening on the given port
    $conn = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($conn) { return Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue }
    return $null
}

function Get-ServerProcess { Get-PortProcess $Port }

function Stop-Server($proc) {
    Write-Host "Stopping server (PID $($proc.Id))..." -ForegroundColor Yellow
    Stop-Process -Id $proc.Id -Force
    # Wait until the port is actually released
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 250
        if (-not (Get-ServerProcess)) { break }
    }
    Write-Host "Server stopped." -ForegroundColor Green
}

function Start-Server {
    Write-Host "Starting dashboard on port $Port..." -ForegroundColor Cyan
    $proc = Start-Process -FilePath 'python' `
        -ArgumentList @("`"$serverScript`"", '--port', $Port) `
        -WorkingDirectory $PSScriptRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError "$logFile.err" `
        -PassThru
    Start-Sleep -Seconds 2
    if ($proc.HasExited) {
        Write-Host "Server failed to start. Last errors:" -ForegroundColor Red
        Get-Content "$logFile.err" -Tail 20
        exit 1
    }
    Write-Host "Server running (PID $($proc.Id)): http://127.0.0.1:$Port" -ForegroundColor Green
}

function Start-Worker {
    if (-not (Test-Path $f5Python)) {
        Write-Host "TTS worker skipped: $f5Python not found." -ForegroundColor Yellow
        return
    }
    Write-Host "Starting TTS worker on port $WorkerPort (loads + compiles models; ~1min warm cache, up to ~5min first ever)..." -ForegroundColor Cyan
    $env:PYTHONUTF8 = '1'
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
    $proc = Start-Process -FilePath $f5Python `
        -ArgumentList @("`"$workerScript`"", '--port', $WorkerPort) `
        -WorkingDirectory $PSScriptRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $workerLog `
        -RedirectStandardError "$workerLog.err" `
        -PassThru
    Start-Sleep -Seconds 2
    if ($proc.HasExited) {
        Write-Host "TTS worker failed to start (TTS will use the slow fallback). Last errors:" -ForegroundColor Yellow
        Get-Content "$workerLog.err" -Tail 20
    } else {
        Write-Host "TTS worker starting (PID $($proc.Id)); serves once models finish loading." -ForegroundColor Green
    }
}

$existing = Get-ServerProcess
$existingWorker = Get-PortProcess $WorkerPort

if ($Stop) {
    if ($existing) { Stop-Server $existing }
    else { Write-Host "No server running on port $Port." -ForegroundColor Yellow }
    if ($existingWorker) {
        Write-Host "Stopping TTS worker (PID $($existingWorker.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $existingWorker.Id -Force
    }
    exit 0
}

if ($existing) {
    Write-Host "Server already running on port $Port (PID $($existing.Id)) - restarting." -ForegroundColor Yellow
    Stop-Server $existing
} else {
    Write-Host "No server running on port $Port." -ForegroundColor Cyan
}
Start-Server

if ($NoWorker) {
    if ($existingWorker) {
        Write-Host "Stopping TTS worker (PID $($existingWorker.Id)) (-NoWorker)..." -ForegroundColor Yellow
        Stop-Process -Id $existingWorker.Id -Force
    }
} else {
    if ($existingWorker) {
        Write-Host "Restarting TTS worker (PID $($existingWorker.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $existingWorker.Id -Force
        Start-Sleep -Milliseconds 500
    }
    Start-Worker
}
