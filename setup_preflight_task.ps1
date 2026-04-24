param(
    [string]$TaskName = "Web_Agent_Preflight_2030",
    [string]$At = "20:30",
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
    param([string]$Name)
    cmd /c "schtasks /Delete /TN `"$Name`" /F >nul 2>nul" | Out-Null
}

function New-InteractiveTask {
    param(
        [string]$Name,
        [string]$Time,
        [string]$TaskCommand
    )
    schtasks /Create /TN $Name /SC DAILY /ST $Time /TR $TaskCommand /RL LIMITED /IT /F | Out-Null
}

function New-PasswordTask {
    param(
        [string]$Name,
        [string]$Time,
        [string]$TaskCommand,
        [string]$UserName,
        [string]$Password
    )
    schtasks /Create /TN $Name /SC DAILY /ST $Time /TR $TaskCommand /RU $UserName /RP $Password /F | Out-Null
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Resolve-PythonExe -Candidate $PythonExe
$runner = Join-Path $root "scheduler_runner.py"
$taskCommand = '"' + $python + '" "' + $runner + '" --doctor --record --self-heal'
$currentUser = if ($RunAsUser) { $RunAsUser } else { [System.Security.Principal.WindowsIdentity]::GetCurrent().Name }

Remove-TaskIfExists -Name $TaskName

$createdMode = ""
if ($RunAsPassword) {
    New-PasswordTask -Name $TaskName -Time $At -TaskCommand $taskCommand -UserName $currentUser -Password $RunAsPassword
    $createdMode = "password"
} else {
    Write-Warning "Creating InteractiveToken preflight task. To enable true offline/background execution, rerun this script or setup_offline_tasks.ps1 with a Windows password."
    New-InteractiveTask -Name $TaskName -Time $At -TaskCommand $taskCommand
    $createdMode = "interactive"
}

Write-Host "Created preflight task using mode: $createdMode"
schtasks /Query /TN $TaskName /FO LIST /V
