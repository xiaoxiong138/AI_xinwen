param(
    [string]$RunAsUser = "",
    [string]$PythonExe = "",
    [string]$NoonTime = "12:00",
    [string]$EveningTime = "21:00",
    [string]$DoctorTime = "09:00",
    [string]$PreflightTime = "20:30"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$currentUser = if ($RunAsUser) { $RunAsUser } else { [System.Security.Principal.WindowsIdentity]::GetCurrent().Name }
$credential = Get-Credential -UserName $currentUser -Message "Enter the Windows password for offline scheduled tasks."
$password = $credential.GetNetworkCredential().Password

& (Join-Path $root "setup_scheduled_tasks.ps1") -RunAsUser $credential.UserName -RunAsPassword $password -PythonExe $PythonExe -NoonTime $NoonTime -EveningTime $EveningTime
& (Join-Path $root "setup_doctor_task.ps1") -RunAsUser $credential.UserName -RunAsPassword $password -PythonExe $PythonExe -At $DoctorTime
& (Join-Path $root "setup_preflight_task.ps1") -RunAsUser $credential.UserName -RunAsPassword $password -PythonExe $PythonExe -At $PreflightTime

Write-Host "Offline-capable scheduled tasks have been created for $($credential.UserName)."
