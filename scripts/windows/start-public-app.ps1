$ErrorActionPreference = "Stop"

$appDirectory = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$streamlitExecutable = Join-Path $appDirectory ".venv\Scripts\streamlit.exe"
$appFile = Join-Path $appDirectory "app.py"
$healthUrl = "http://127.0.0.1:8501/_stcore/health"
$publicUrl = "https://cazettes-lab.com"
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
    Write-Host "Starting the public tunnel..." -ForegroundColor Cyan
    try {
        Start-Service -Name "cloudflared" -ErrorAction Stop
    } catch {
        # Elevate only the service operation; Streamlit stays under the current user.
        $serviceCommand = "Start-Service -Name 'cloudflared' -ErrorAction Stop"
        $encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($serviceCommand))
        $serviceStarter = Start-Process -FilePath "powershell.exe" `
            -ArgumentList @("-NoProfile", "-EncodedCommand", $encodedCommand) `
            -Verb RunAs -WindowStyle Hidden -Wait -PassThru
        if ($serviceStarter.ExitCode -ne 0) {
            throw "The public tunnel could not be started."
        }
    }
    $tunnelService = Get-Service -Name "cloudflared"
    $tunnelService.WaitForStatus("Running", [TimeSpan]::FromSeconds(20))
    Write-Host "Public address: $publicUrl" -ForegroundColor Cyan
} else {
    Write-Host "Cloudflare tunnel service is running." -ForegroundColor Green
    Write-Host "Public address: $publicUrl" -ForegroundColor Cyan
}

if ($null -ne $tunnelService -and (Get-Service -Name "cloudflared").Status -eq "Running") {
    Write-Host "Waiting for the public address..." -ForegroundColor Cyan
    $publicReady = $false
    for ($attempt = 0; $attempt -lt 10; $attempt++) {
        try {
            $publicHealth = Invoke-WebRequest -Uri "$publicUrl/_stcore/health" -UseBasicParsing -TimeoutSec 3
            if ($publicHealth.StatusCode -eq 200 -and $publicHealth.Content.Trim() -eq "ok") {
                $publicReady = $true
                break
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $publicReady) {
        throw "The app is running locally, but the public tunnel is not ready. Check logs\cloudflared.log."
    }
    Start-Process -FilePath $publicUrl
} else {
    Start-Process -FilePath "http://127.0.0.1:8501"
}

