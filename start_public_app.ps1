$ErrorActionPreference = "Stop"

$appDirectory = $PSScriptRoot
$streamlitExecutable = Join-Path $appDirectory ".venv\Scripts\streamlit.exe"
$appFile = Join-Path $appDirectory "app.py"
$cloudflaredExecutable = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
$healthUrl = "http://127.0.0.1:8501/_stcore/health"
$logDirectory = Join-Path $appDirectory "logs"

if (-not (Test-Path -LiteralPath $streamlitExecutable)) {
    throw "Streamlit executable not found: $streamlitExecutable"
}
if (-not (Test-Path -LiteralPath $cloudflaredExecutable)) {
    throw "cloudflared executable not found: $cloudflaredExecutable"
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
Write-Host "Creating the public Cloudflare URL..." -ForegroundColor Cyan
Write-Host "Keep this window open. Press Ctrl+C to stop public access." -ForegroundColor DarkGray
Write-Host ""

& $cloudflaredExecutable tunnel `
    --url "http://127.0.0.1:8501" `
    --protocol http2 `
    --no-autoupdate

