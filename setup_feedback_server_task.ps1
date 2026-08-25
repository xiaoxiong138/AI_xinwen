param(
    [string]$TaskName = "Web_Agent_Feedback_Server",
    [string]$PythonExe = "",
    [string]$ProjectDir = "D:\Web_Agent",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 18765
)

$ErrorActionPreference = "Stop"

if (-not $PythonExe) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $PythonExe = $pythonCommand.Source
    } else {
        $PythonExe = "python"
    }
}

$scriptPath = Join-Path $ProjectDir "feedback_server.py"
if (-not (Test-Path $scriptPath)) {
    throw "feedback_server.py not found at $scriptPath"
}

$action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "`"$scriptPath`" --host $HostAddress --port $Port" `
    -WorkingDirectory $ProjectDir

$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Description "Run local Web_Agent email feedback service at user logon." `
        -Force | Out-Null

    Start-ScheduledTask -TaskName $TaskName
    Write-Host "Feedback server task installed and started: $TaskName"
} catch {
    Write-Warning "Register-ScheduledTask failed; trying schtasks current-user logon task."
    $taskCommand = "`"$PythonExe`" `"$scriptPath`" --host $HostAddress --port $Port"
    & schtasks.exe /Create /TN $TaskName /SC ONLOGON /TR $taskCommand /RL LIMITED /F | Out-Null
    if ($LASTEXITCODE -eq 0) {
        & schtasks.exe /Run /TN $TaskName | Out-Null
        Write-Host "Feedback server task installed with schtasks and started: $TaskName"
    } else {
        $startupDir = [Environment]::GetFolderPath("Startup")
        $startupCmd = Join-Path $startupDir "$TaskName.cmd"
        $cmd = "@echo off`r`ncd /d `"$ProjectDir`"`r`nstart `"`" /min `"$PythonExe`" `"$scriptPath`" --host $HostAddress --port $Port`r`n"
        Set-Content -Path $startupCmd -Value $cmd -Encoding ASCII
        Start-Process -FilePath $PythonExe -ArgumentList "`"$scriptPath`" --host $HostAddress --port $Port" -WorkingDirectory $ProjectDir -WindowStyle Hidden
        Write-Warning "schtasks fallback failed; installed current-user startup command instead: $startupCmd"
    }
}

Write-Host "Endpoint: http://$HostAddress`:$Port/"
