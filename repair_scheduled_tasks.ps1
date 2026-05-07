param(
    [string]$RunAsUser = "",
    [string]$PythonExe = "",
    [string]$NoonTime = "12:00",
    [string]$EveningTime = "21:00",
    [string]$DoctorTime = "09:00",
    [string]$PreflightTime = "20:30",
    [string]$RunAsPassword = "",
    [switch]$UseS4U,
    [switch]$NoPrompt
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()

function Remove-TaskIfExists {
    param([string]$TaskName)
    if (-not $TaskName) {
        return
    }
    cmd /c "schtasks /Delete /TN `"$TaskName`" /F >nul 2>nul" | Out-Null
}

function Export-TaskBackupIfExists {
    param(
        [string]$TaskName,
        [string]$BackupRoot
    )
    if (-not $TaskName) {
        return
    }
    $safeName = $TaskName.Replace("\", "_").Replace("/", "_")
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupPath = Join-Path $BackupRoot "$safeName`_$stamp.xml"
    $xml = schtasks /Query /TN $TaskName /XML 2>$null
    if ($LASTEXITCODE -eq 0 -and $xml) {
        $xml | Out-File -FilePath $backupPath -Encoding UTF8
        Write-Host "Backed up $TaskName to $backupPath"
    }
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backupRoot = Join-Path $root "logs\task_backups"
New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
$currentUser = if ($RunAsUser) { $RunAsUser } else { [System.Security.Principal.WindowsIdentity]::GetCurrent().Name }
$password = $RunAsPassword
$credentialUserName = $currentUser

Write-Host "Removing legacy Web_Agent scheduled tasks..."
Export-TaskBackupIfExists -TaskName "Web_Agent_Send_1200" -BackupRoot $backupRoot
Export-TaskBackupIfExists -TaskName "Web_Agent_Send_2100" -BackupRoot $backupRoot
Remove-TaskIfExists -TaskName "Web_Agent_Send_1200"
Remove-TaskIfExists -TaskName "Web_Agent_Send_2100"

if (-not $UseS4U -and -not $password) {
    $envPassword = [Environment]::GetEnvironmentVariable("WEB_AGENT_RUNAS_PASSWORD")
    if ($envPassword) {
        $password = $envPassword
    }
}

$envUser = [Environment]::GetEnvironmentVariable("WEB_AGENT_RUNAS_USER")
if (-not $RunAsUser -and $envUser) {
    $credentialUserName = $envUser
}

if (-not $UseS4U -and -not $password) {
    if ($NoPrompt) {
        throw "Missing Windows password. Set WEB_AGENT_RUNAS_PASSWORD or rerun without -NoPrompt."
    }
    $credential = Get-Credential -UserName $credentialUserName -Message "Enter the Windows password for offline Web_Agent scheduled tasks."
    $credentialUserName = $credential.UserName
    $password = $credential.GetNetworkCredential().Password
}

if ($UseS4U) {
    Write-Host "Rebuilding S4U/background Web_Agent tasks for $credentialUserName..."
    & (Join-Path $root "setup_offline_tasks.ps1") `
        -RunAsUser $credentialUserName `
        -UseS4U `
        -PythonExe $PythonExe `
        -NoonTime $NoonTime `
        -EveningTime $EveningTime `
        -DoctorTime $DoctorTime `
        -PreflightTime $PreflightTime
} else {
    Write-Host "Rebuilding offline-capable Web_Agent tasks for $credentialUserName..."
    & (Join-Path $root "setup_offline_tasks.ps1") `
        -RunAsUser $credentialUserName `
        -RunAsPassword $password `
        -PythonExe $PythonExe `
        -NoonTime $NoonTime `
        -EveningTime $EveningTime `
        -DoctorTime $DoctorTime `
        -PreflightTime $PreflightTime
}

$python = if ($PythonExe) { $PythonExe } elseif (Test-Path "D:\python\python.exe") { "D:\python\python.exe" } else { "python" }
Write-Host "Recording fresh doctor snapshot..."
& $python (Join-Path $root "scheduler_runner.py") --doctor --record

Write-Host "Scheduled task repair completed. Check logs\doctor_latest.json for the latest health snapshot."
