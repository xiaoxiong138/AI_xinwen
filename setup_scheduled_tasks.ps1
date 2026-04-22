param(
    [string]$TaskNameNoon = "Web_Agent_Send_1200_v2",
    [string]$TaskNameEvening = "Web_Agent_Send_2100_v2",
    [string]$NoonTime = "12:00",
    [string]$EveningTime = "21:00",
    [string]$PythonExe = "",
    [string]$RunAsUser = "",
    [string]$RunAsPassword = ""
)

$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
    param([string]$Candidate)
    if ($Candidate) {
        return $Candidate
    }
    if (Test-Path "D:\python\python.exe") {
        return "D:\python\python.exe"
    }
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    throw "Python executable not found."
}

function Remove-TaskIfExists {
    param([string]$TaskName)
    cmd /c "schtasks /Delete /TN `"$TaskName`" /F >nul 2>nul" | Out-Null
}

function New-InteractiveTask {
    param(
        [string]$TaskName,
        [string]$At,
        [string]$TaskCommand
    )
    schtasks /Create /TN $TaskName /SC DAILY /ST $At /TR $TaskCommand /RL LIMITED /IT /F | Out-Null
}

function New-PasswordTask {
    param(
        [string]$TaskName,
        [string]$At,
        [string]$TaskCommand,
        [string]$UserName,
        [string]$Password
    )
    schtasks /Create /TN $TaskName /SC DAILY /ST $At /TR $TaskCommand /RU $UserName /RP $Password /F | Out-Null
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Resolve-PythonExe -Candidate $PythonExe
$runner = Join-Path $root "scheduler_runner.py"
$taskCommand = '"' + $python + '" "' + $runner + '"'
$currentUser = if ($RunAsUser) { $RunAsUser } else { [System.Security.Principal.WindowsIdentity]::GetCurrent().Name }

Remove-TaskIfExists -TaskName $TaskNameNoon
Remove-TaskIfExists -TaskName $TaskNameEvening

$createdMode = ""
if ($RunAsPassword) {
    New-PasswordTask -TaskName $TaskNameNoon -At $NoonTime -TaskCommand $taskCommand -UserName $currentUser -Password $RunAsPassword
    New-PasswordTask -TaskName $TaskNameEvening -At $EveningTime -TaskCommand $taskCommand -UserName $currentUser -Password $RunAsPassword
    $createdMode = "password"
} else {
    Write-Warning "Creating InteractiveToken tasks because S4U/background mode breaks network access for this project. To enable true offline/background sending, rerun this script with -RunAsPassword."
    New-InteractiveTask -TaskName $TaskNameNoon -At $NoonTime -TaskCommand $taskCommand
    New-InteractiveTask -TaskName $TaskNameEvening -At $EveningTime -TaskCommand $taskCommand
    $createdMode = "interactive"
}

Write-Host "Created tasks using mode: $createdMode"
schtasks /Query /TN $TaskNameNoon /FO LIST /V
Write-Host "---"
schtasks /Query /TN $TaskNameEvening /FO LIST /V
