param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]]$ReportPath,
    [string]$OutputDir = "artifacts/email_ui_audit"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$bundledRoot = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies"
$nodeExe = if ($env:CODEX_NODE_EXE) {
    $env:CODEX_NODE_EXE
} elseif (Test-Path (Join-Path $bundledRoot "node\bin\node.exe")) {
    Join-Path $bundledRoot "node\bin\node.exe"
} else {
    (Get-Command node -ErrorAction Stop).Source
}

if (-not $env:NODE_PATH) {
    $env:NODE_PATH = Join-Path $bundledRoot "node\node_modules"
}

$resolvedReports = @(
    foreach ($value in $ReportPath) {
        $resolved = (Resolve-Path -LiteralPath $value -ErrorAction Stop).Path
        $item = Get-Item -LiteralPath $resolved -ErrorAction Stop
        if (-not $item.PSIsContainer -and $item.Extension -in @(".html", ".htm")) {
            $resolved
            continue
        }
        throw "Email UI audit input must be an HTML file: $resolved"
    }
)
$resolvedOutput = if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
} else {
    Join-Path $root $OutputDir
}

& $nodeExe (Join-Path $PSScriptRoot "email_ui_audit.js") --output-dir $resolvedOutput @resolvedReports
exit $LASTEXITCODE
