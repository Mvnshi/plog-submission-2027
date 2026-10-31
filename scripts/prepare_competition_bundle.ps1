param(
    [Parameter(Mandatory=$true)][string]$SourceDirectory,
    [Parameter(Mandatory=$true)][string]$DestinationDirectory
)

# Preparation-only convenience entry point. The root workflow performs public preflight first.
& (Join-Path (Split-Path -Parent $PSScriptRoot) 'run_first_contact.ps1') -SourceDirectory $SourceDirectory -DestinationDirectory $DestinationDirectory -PrepareOnly
exit $LASTEXITCODE
