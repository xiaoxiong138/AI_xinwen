param(
    [string]$TaskName = "Web_Agent_Doctor_0900",
    [string]$At = "09:00",
    [string]$PythonExe = "",
    [string]$RunAsUser = "",
    [string]$RunAsPassword = "",
    [switch]$UseS4U
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()

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

function Invoke-Schtasks {
    param([string[]]$Arguments)
    & schtasks @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "schtasks failed with exit code ${LASTEXITCODE}: schtasks $($Arguments -join ' ')"
    }
}

function New-InteractiveTask {
    param(
        [string]$Name,
        [string]$Time,
        [string]$TaskCommand
    )
    Invoke-Schtasks -Arguments @("/Create", "/TN", $Name, "/SC", "DAILY", "/ST", $Time, "/TR", $TaskCommand, "/RL", "LIMITED", "/IT", "/F")
}

function New-PasswordTask {
    param(
        [string]$Name,
        [string]$Time,
        [string]$TaskCommand,
        [string]$UserName,
        [string]$Password
    )
    Invoke-Schtasks -Arguments @("/Create", "/TN", $Name, "/SC", "DAILY", "/ST", $Time, "/TR", $TaskCommand, "/RU", $UserName, "/RP", $Password, "/F")
}

function New-S4UTask {
    param(
        [string]$Name,
        [string]$Time,
        [string]$TaskCommand,
        [string]$UserName
    )
    Invoke-Schtasks -Arguments @("/Create", "/TN", $Name, "/SC", "DAILY", "/ST", $Time, "/TR", $TaskCommand, "/RU", $UserName, "/NP", "/RL", "LIMITED", "/F")
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Resolve-PythonExe -Candidate $PythonExe
$runner = Join-Path $root "scheduler_runner.py"
$taskCommand = '"' + $python + '" "' + $runner + '" --doctor --record'
$currentUser = if ($RunAsUser) { $RunAsUser } else { [System.Security.Principal.WindowsIdentity]::GetCurrent().Name }

Remove-TaskIfExists -Name $TaskName

$createdMode = ""
if ($RunAsPassword) {
    New-PasswordTask -Name $TaskName -Time $At -TaskCommand $taskCommand -UserName $currentUser -Password $RunAsPassword
    $createdMode = "password"
} elseif ($UseS4U) {
    Write-Warning "Creating S4U/background doctor task without storing a password."
    New-S4UTask -Name $TaskName -Time $At -TaskCommand $taskCommand -UserName $currentUser
    $createdMode = "s4u"
} else {
    Write-Warning "Creating InteractiveToken doctor task. Rerun with -UseS4U to avoid InteractiveToken-only failures, or with -RunAsPassword for strongest offline/background execution."
    New-InteractiveTask -Name $TaskName -Time $At -TaskCommand $taskCommand
    $createdMode = "interactive"
}

Write-Host "Created doctor task using mode: $createdMode"
schtasks /Query /TN $TaskName /FO LIST /V
