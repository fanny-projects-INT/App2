$ErrorActionPreference = "Stop"

$appDirectory = $PSScriptRoot
$streamlitExecutable = Join-Path $appDirectory ".venv\Scripts\streamlit.exe"
$appFile = Join-Path $appDirectory "app.py"
$healthUrl = "http://127.0.0.1:8501/_stcore/health"
$publicUrl = "https://cazette-lab.org"
$logDirectory = Join-Path $appDirectory "logs"

if (-not (Test-Path -LiteralPath $streamlitExecutable)) {
    throw "Streamlit executable not found: $streamlitExecutable"
}
if (-not (Test-Path -LiteralPath $logDirectory)) {
    New-Item -ItemType Directory -Path $logDirectory | Out-Null
}

$streamlitReady = $false
try {
    $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
    $streamlitReady = $response.StatusCode -eq 200
} catch {
    $streamlitReady = $false
}

if (-not $streamlitReady) {
    Write-Host "Starting Cazette dashboard..." -ForegroundColor Cyan
    $stdoutLog = Join-Path $logDirectory "streamlit.out.log"
    $stderrLog = Join-Path $logDirectory "streamlit.err.log"
    Start-Process `
        -FilePath $streamlitExecutable `
        -ArgumentList @(
            "run", $appFile,
            "--server.address", "127.0.0.1",
            "--server.port", "8501",
            "--server.headless", "true"
        ) `
        -WorkingDirectory $appDirectory `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog

    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Seconds 1
        try {
            $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                $streamlitReady = $true
                break
            }
        } catch {
            $streamlitReady = $false
        }
    }
}

if (-not $streamlitReady) {
    throw "Streamlit did not start. Check logs\streamlit.err.log."
}

Write-Host "Dashboard ready." -ForegroundColor Green
$tunnelService = Get-Service -Name "cloudflared" -ErrorAction SilentlyContinue
if ($null -eq $tunnelService) {
    Write-Warning "The permanent Cloudflare tunnel service is not installed yet."
    Write-Host "The dashboard is available locally at http://127.0.0.1:8501" -ForegroundColor Cyan
} elseif ($tunnelService.Status -ne "Running") {
    Write-Warning "The Cloudflare tunnel service is installed but is not running."
} else {
    Write-Host "Cloudflare tunnel service is running." -ForegroundColor Green
    Write-Host "Public address: $publicUrl" -ForegroundColor Cyan
}

