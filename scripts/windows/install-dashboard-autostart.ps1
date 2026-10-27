$ErrorActionPreference = "Stop"

$taskName = "Cazette Dashboard"
$appDirectory = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$launcher = Join-Path $appDirectory "scripts\windows\start-public-app.ps1"

if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Dashboard launcher not found: $launcher"
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$launcher`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Starts the Cazette Streamlit dashboard on local port 8501." `
    -Force | Out-Null

Write-Host "Automatic startup configured: $taskName" -ForegroundColor Green
