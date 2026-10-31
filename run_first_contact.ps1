param(
    [Parameter(Mandatory=$true)][string]$SourceDirectory,
    [string]$DestinationDirectory = (Join-Path $env:USERPROFILE 'PLOG-licensed'),
    [switch]$PrepareOnly
)

# Operator-only local execution. Hosted agents may use this ONLY with generated synthetic inputs.
$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
$pythonPath = Join-Path $projectDirectory '.venv/Scripts/python.exe'
try {
    # Resolve arguments before changing location; square brackets are literal, not wildcards.
    $sourcePath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($SourceDirectory)
    $destinationPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($DestinationDirectory)
    Push-Location -LiteralPath $projectDirectory
    try {
        & (Join-Path $projectDirectory 'run_competition.ps1') -PreflightOnly *> $null
        if ($LASTEXITCODE -ne 0) { throw 'Public release preflight failed.' }
        $arguments = @('-m','plog.bundle','--source',$sourcePath,'--destination',$destinationPath)
        if (-not $PrepareOnly) { $arguments += '--run' }
        & $pythonPath @arguments
        $resultCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
} catch {
    [Console]::Error.WriteLine('PLOG FIRST CONTACT FAILED: setup could not start. Check the pinned installation and local filesystem paths; keep diagnostics local.')
    exit 1
}
exit $resultCode
