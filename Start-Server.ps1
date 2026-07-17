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
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
$serverScript = Join-Path $PSScriptRoot 'webui\server.py'
$logFile      = Join-Path $PSScriptRoot 'webui\server.log'

function Get-ServerProcess {
    # Find the process listening on the dashboard port
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($conn) { return Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue }
    return $null
}

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

$existing = Get-ServerProcess

if ($Stop) {
    if ($existing) { Stop-Server $existing }
    else { Write-Host "No server running on port $Port." -ForegroundColor Yellow }
    exit 0
}

if ($existing) {
    Write-Host "Server already running on port $Port (PID $($existing.Id)) - restarting." -ForegroundColor Yellow
    Stop-Server $existing
} else {
    Write-Host "No server running on port $Port." -ForegroundColor Cyan
}
Start-Server
