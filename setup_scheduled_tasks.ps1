param(
    [string]$TaskNameNoon = "Web_Agent_Send_1300_v2",
    [string]$TaskNameEvening = "Web_Agent_Send_2100_v2",
    [string]$NoonTime = "13:00",
    [string]$EveningTime = "21:00",
    [string]$PythonExe = "",
    [string]$RunAsUser = "",
    [string]$RunAsPassword = "",
    [switch]$UseS4U,
    [string[]]$LegacyTaskNames = @("Web_Agent_Send_1200", "Web_Agent_Send_1200_v2", "Web_Agent_Send_2100")
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

function Optimize-TaskRuntimeSettings {
    param(
        [string]$TaskName,
        [string]$PythonPath,
        [string]$RunnerPath,
        [string]$WorkingDirectory,
        [string]$HiddenLauncherPath
    )
    if (-not (Get-Command Get-ScheduledTask -ErrorAction SilentlyContinue) -or
        -not (Get-Command Set-ScheduledTask -ErrorAction SilentlyContinue) -or
        -not (Get-Command New-ScheduledTaskAction -ErrorAction SilentlyContinue)) {
        Write-Warning "ScheduledTasks module is unavailable; runtime settings were not optimized for $TaskName."
        return
    }

    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    $task.Settings.DisallowStartIfOnBatteries = $false
    $task.Settings.StopIfGoingOnBatteries = $false
    $task.Settings.StartWhenAvailable = $true
    $task.Settings.ExecutionTimeLimit = "PT2H"
    $task.Settings.RestartCount = 3
    $task.Settings.RestartInterval = "PT5M"
    $task.Settings.IdleSettings.StopOnIdleEnd = $false
    $task.Settings.IdleSettings.RestartOnIdle = $false
    $action = New-ScheduledTaskAction `
        -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
        -Argument ('-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $HiddenLauncherPath + '" -PythonExe "' + $PythonPath + '" -RunnerPath "' + $RunnerPath + '" -WorkingDirectory "' + $WorkingDirectory + '"') `
        -WorkingDirectory $WorkingDirectory
    Set-ScheduledTask -TaskName $TaskName -Action $action -Settings $task.Settings | Out-Null
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Resolve-PythonExe -Candidate $PythonExe
$runner = Join-Path $root "scheduler_runner.py"
$hiddenLauncher = Join-Path $root "run_scheduler_hidden.ps1"
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Scheduler runner not found: $runner"
}
if (-not (Test-Path -LiteralPath $hiddenLauncher -PathType Leaf)) {
    throw "Hidden scheduler launcher not found: $hiddenLauncher"
}
$taskCommand = 'powershell.exe -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $hiddenLauncher + '" -PythonExe "' + $python + '"'
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

Optimize-TaskRuntimeSettings -TaskName $TaskNameNoon -PythonPath $python -RunnerPath $runner -WorkingDirectory $root -HiddenLauncherPath $hiddenLauncher
Optimize-TaskRuntimeSettings -TaskName $TaskNameEvening -PythonPath $python -RunnerPath $runner -WorkingDirectory $root -HiddenLauncherPath $hiddenLauncher

Write-Host "Created tasks using mode: $createdMode"
schtasks /Query /TN $TaskNameNoon /FO LIST /V
Write-Host "---"
schtasks /Query /TN $TaskNameEvening /FO LIST /V
