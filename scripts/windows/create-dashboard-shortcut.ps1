$ErrorActionPreference = "Stop"

$appDirectory = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$desktopDirectory = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopDirectory "Cazette Dashboard.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $env:ComSpec
$shortcut.Arguments = '/c ""' + (Join-Path $appDirectory "START_PUBLIC_APP.cmd") + '""'
$shortcut.WorkingDirectory = $appDirectory
$shortcut.Description = "Start the Cazette dashboard and public tunnel, then open the website."
$shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,13"
$shortcut.WindowStyle = 7
$shortcut.Save()
Write-Host "Shortcut created: $shortcutPath"
