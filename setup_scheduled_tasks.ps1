param(
    [string]$TaskNameNoon = "Web_Agent_Send_1200_v2",
    [string]$TaskNameEvening = "Web_Agent_Send_2100_v2",
    [string]$NoonTime = "12:00",
    [string]$EveningTime = "21:00",
    [string]$PythonExe = "",
    [string]$RunAsUser = "",
    [string]$RunAsPassword = "",
    [switch]$UseS4U,
    [string[]]$LegacyTaskNames = @("Web_Agent_Send_1200", "Web_Agent_Send_2100")
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
    param([string]$TaskName)
    cmd /c "schtasks /Delete /TN `"$TaskName`" /F >nul 2>nul" | Out-Null
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
        [string]$TaskName,
        [string]$At,
        [string]$TaskCommand
    )
    Invoke-Schtasks -Arguments @("/Create", "/TN", $TaskName, "/SC", "DAILY", "/ST", $At, "/TR", $TaskCommand, "/RL", "LIMITED", "/IT", "/F")
}

function New-PasswordTask {
    param(
        [string]$TaskName,
        [string]$At,
        [string]$TaskCommand,
        [string]$UserName,
        [string]$Password
    )
    Invoke-Schtasks -Arguments @("/Create", "/TN", $TaskName, "/SC", "DAILY", "/ST", $At, "/TR", $TaskCommand, "/RU", $UserName, "/RP", $Password, "/F")
}

function New-S4UTask {
    param(
        [string]$TaskName,
        [string]$At,
        [string]$TaskCommand,
        [string]$UserName
    )
    Invoke-Schtasks -Arguments @("/Create", "/TN", $TaskName, "/SC", "DAILY", "/ST", $At, "/TR", $TaskCommand, "/RU", $UserName, "/NP", "/RL", "LIMITED", "/F")
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Resolve-PythonExe -Candidate $PythonExe
$runner = Join-Path $root "scheduler_runner.py"
$taskCommand = '"' + $python + '" "' + $runner + '"'
$currentUser = if ($RunAsUser) { $RunAsUser } else { [System.Security.Principal.WindowsIdentity]::GetCurrent().Name }

Remove-TaskIfExists -TaskName $TaskNameNoon
Remove-TaskIfExists -TaskName $TaskNameEvening
foreach ($legacyTaskName in $LegacyTaskNames) {
    if ($legacyTaskName -and $legacyTaskName -notin @($TaskNameNoon, $TaskNameEvening)) {
        Remove-TaskIfExists -TaskName $legacyTaskName
    }
}

$createdMode = ""
if ($RunAsPassword) {
    New-PasswordTask -TaskName $TaskNameNoon -At $NoonTime -TaskCommand $taskCommand -UserName $currentUser -Password $RunAsPassword
    New-PasswordTask -TaskName $TaskNameEvening -At $EveningTime -TaskCommand $taskCommand -UserName $currentUser -Password $RunAsPassword
    $createdMode = "password"
} elseif ($UseS4U) {
    Write-Warning "Creating S4U/background tasks without storing a password. This is less offline-capable than password mode, but avoids InteractiveToken-only failures."
    New-S4UTask -TaskName $TaskNameNoon -At $NoonTime -TaskCommand $taskCommand -UserName $currentUser
    New-S4UTask -TaskName $TaskNameEvening -At $EveningTime -TaskCommand $taskCommand -UserName $currentUser
    $createdMode = "s4u"
} else {
    Write-Warning "Creating InteractiveToken tasks. To avoid InteractiveToken-only failures without a Windows password, rerun this script with -UseS4U. For strongest offline/background sending, rerun with -RunAsPassword."
    New-InteractiveTask -TaskName $TaskNameNoon -At $NoonTime -TaskCommand $taskCommand
    New-InteractiveTask -TaskName $TaskNameEvening -At $EveningTime -TaskCommand $taskCommand
    $createdMode = "interactive"
}

Write-Host "Created tasks using mode: $createdMode"
schtasks /Query /TN $TaskNameNoon /FO LIST /V
Write-Host "---"
schtasks /Query /TN $TaskNameEvening /FO LIST /V
