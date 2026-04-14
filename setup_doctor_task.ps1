param(
    [string]$TaskName = "Web_Agent_Doctor_0900",
    [string]$At = "09:00",
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
    schtasks /Create /TN $Name /SC DAILY /ST $Time /TR $TaskCommand /F | Out-Null
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

function New-S4UTask {
    param(
        [string]$Name,
        [datetime]$Time,
        [string]$PythonPath,
        [string]$RunnerPath,
        [string]$UserName
    )
    $action = New-ScheduledTaskAction -Execute $PythonPath -Argument ('"' + $RunnerPath + '" --doctor --record')
    $trigger = New-ScheduledTaskTrigger -Daily -At $Time
    $principal = New-ScheduledTaskPrincipal -UserId $UserName -LogonType S4U
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 1)
    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
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
} else {
    try {
        New-S4UTask -Name $TaskName -Time ([datetime]::Parse($At)) -PythonPath $python -RunnerPath $runner -UserName $currentUser
        $createdMode = "s4u"
    } catch {
        Write-Warning "Non-interactive doctor task creation was blocked. Falling back to interactive mode. To force offline/background execution, rerun this script in an elevated PowerShell or provide -RunAsPassword."
        Remove-TaskIfExists -Name $TaskName
        New-InteractiveTask -Name $TaskName -Time $At -TaskCommand $taskCommand
        $createdMode = "interactive"
    }
}

Write-Host "Created doctor task using mode: $createdMode"
schtasks /Query /TN $TaskName /FO LIST /V
