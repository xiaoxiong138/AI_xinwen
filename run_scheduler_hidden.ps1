param(
    [string]$PythonExe = "D:\python\python.exe",
    [string]$RunnerPath = "",
    [string]$WorkingDirectory = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArguments = @()
)

$ErrorActionPreference = "Stop"
if (-not $WorkingDirectory) {
    $WorkingDirectory = $PSScriptRoot
}
if (-not $RunnerPath) {
    $RunnerPath = Join-Path $WorkingDirectory "scheduler_runner.py"
}

$logDirectory = Join-Path $WorkingDirectory "logs"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$startedAt = Get-Date
$launcherLogPath = Join-Path $logDirectory ("scheduler_launcher_{0}_{1}.log" -f $startedAt.ToString("yyyyMMdd_HHmmss"), $PID)
$launcherStatusPath = Join-Path $logDirectory "scheduler_launcher_latest.json"
Get-ChildItem -LiteralPath $logDirectory -Filter "scheduler_launcher_latest.json.tmp.*" -File -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-1) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

function Write-LauncherLog {
    param([string]$Message)
    Add-Content -LiteralPath $launcherLogPath -Value ("[{0}] {1}" -f (Get-Date).ToString("s"), $Message) -Encoding UTF8
}

function Write-LauncherStatus {
    param(
        [string]$Status,
        [int]$ExitCode,
        [string]$ErrorMessage = ""
    )
    $payload = [ordered]@{
        status = $Status
        pid = $PID
        started_at = $startedAt.ToString("s")
        finished_at = if ($Status -eq "running") { "" } else { (Get-Date).ToString("s") }
        exit_code = $ExitCode
        python_exe = $PythonExe
        runner_path = $RunnerPath
        working_directory = $WorkingDirectory
        log_file = $launcherLogPath
        error = $ErrorMessage
    }
    $temporaryStatusPath = $launcherStatusPath + ".tmp.$PID"
    $payload | ConvertTo-Json | Set-Content -LiteralPath $temporaryStatusPath -Encoding UTF8
    Move-Item -LiteralPath $temporaryStatusPath -Destination $launcherStatusPath -Force
}

try {
    Write-LauncherLog "Launcher started."
    Write-LauncherStatus -Status "running" -ExitCode 0
    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
        throw "Python executable not found: $PythonExe"
    }
    if (-not (Test-Path -LiteralPath $RunnerPath -PathType Leaf)) {
        throw "Scheduler runner not found: $RunnerPath"
    }
    $workspaceDrive = Get-PSDrive -Name ([System.IO.Path]::GetPathRoot($WorkingDirectory).TrimEnd('\').TrimEnd(':'))
    $freeDiskMb = [math]::Round($workspaceDrive.Free / 1MB, 1)
    Write-LauncherLog "Workspace disk free: $freeDiskMb MB."
    if ($freeDiskMb -lt 8) {
        throw "Workspace disk critically low: $freeDiskMb MB free; at least 8 MB is required."
    }
    Set-Location -LiteralPath $WorkingDirectory
    & $PythonExe $RunnerPath @RunnerArguments
    $runnerExitCode = [int]$LASTEXITCODE
    $finalStatus = if ($runnerExitCode -eq 0) { "success" } else { "failed" }
    Write-LauncherLog "Launcher finished with runner exit code $runnerExitCode."
    Write-LauncherStatus -Status $finalStatus -ExitCode $runnerExitCode
    exit $runnerExitCode
} catch {
    $errorMessage = $_.Exception.Message
    Write-LauncherLog "Launcher failed: $errorMessage"
    Write-LauncherStatus -Status "exception" -ExitCode 1 -ErrorMessage $errorMessage
    exit 1
}
