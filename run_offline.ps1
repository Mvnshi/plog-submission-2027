param(
    [Parameter(Mandatory=$true)][string]$DataDir,
    [string]$OutputPath = 'plog.csv',
    [string]$ConfigPath = 'configs/default.json',
    [string]$ModelPath = '',
    [switch]$UseSmartTagging
)

# Run on the licensed machine outside any hosted coding-agent session. No network calls.
$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
$pythonPath = Join-Path $projectDirectory '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Install the pinned environment first; see RUNBOOK.md. This script makes no network calls.'
}
Push-Location -LiteralPath $projectDirectory
try {
    $resolvedOutput = [IO.Path]::GetFullPath($OutputPath)
    $replicaPath = $resolvedOutput + '.regenerated.csv'
    $inferArguments = @('-m','plog','offline','--data-dir',$DataDir,'--config',$ConfigPath)
    if ($ModelPath) { $inferArguments += @('--model',$ModelPath) }
    if ($UseSmartTagging) { $inferArguments += '--events' }
    & $pythonPath @inferArguments --out $resolvedOutput
    if ($LASTEXITCODE -ne 0) { throw 'Initial inference or official validation failed.' }
    & $pythonPath @inferArguments --out $replicaPath --audit-reference $resolvedOutput
    if ($LASTEXITCODE -ne 0) { throw 'Independent regeneration or byte comparison failed.' }
    & $pythonPath evaluation/validate_submission.py $resolvedOutput
    if ($LASTEXITCODE -ne 0) { throw 'Final official validation failed.' }
    & $pythonPath -m plog audit $resolvedOutput $replicaPath
    if ($LASTEXITCODE -ne 0) { throw 'Final byte audit failed.' }
    Write-Output 'Local inference, official validation, and independent byte-for-byte regeneration passed.'
} finally {
    Pop-Location
}
