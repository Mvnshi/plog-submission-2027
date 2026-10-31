param(
    [string]$DataDir = '',
    [string]$OutputDirectory = '',
    [ValidateSet('release','challenger')][string]$Candidate = 'release',
    [switch]$PreflightOnly
)

# Run with licensed inputs only outside hosted-agent sessions. This script makes no network calls.
# The release allowlist is explicit. Rejected experimental models cannot be selected by filename.
$ErrorActionPreference = 'Stop'
$projectDirectory = [IO.Path]::GetFullPath($PSScriptRoot)
$invocationDirectory = $ExecutionContext.SessionState.Path.CurrentFileSystemLocation.ProviderPath
if ($Candidate -ne 'release') {
    throw 'No separately validated challenger is approved for competition execution. Use release.'
}
$configPath = Join-Path $projectDirectory 'configs/selected.json'
$modelPath = Join-Path $projectDirectory 'models/ball_free_sportec_v1.json'
$pythonPath = Join-Path $projectDirectory '.venv/Scripts/python.exe'
$offlineScript = Join-Path $projectDirectory 'run_offline.ps1'
$runtimeManifest = Join-Path $projectDirectory 'release/frozen-runtime.json'
$expectedConfigHash = '62870ec2017962f73ee7e5972176461698510bb9c94599aacaeb9039293c05d3'
$expectedModelHash = 'd41d788cf7ce43b16f8a6b81a8dd931849cdb44a282e9874762c25fa79c57971'
function Get-LocalSHA256([string]$Path) {
    # Use .NET directly: Get-FileHash is absent on some otherwise usable Windows shells.
    $stream = [IO.File]::OpenRead($Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-','').ToLowerInvariant()
    } finally {
        $algorithm.Dispose()
        $stream.Dispose()
    }
}
function Get-CallerFileSystemPath([string]$Path) {
    # Preflight changes location for public-source checks. Operator paths retain the
    # caller's original filesystem location, including drive-relative paths.
    Push-Location -LiteralPath $invocationDirectory
    try {
        $provider = $null
        $drive = $null
        $resolved = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath(
            $Path, [ref]$provider, [ref]$drive)
        if ($provider.Name -ne 'FileSystem') { throw 'A filesystem path is required.' }
        return $resolved
    } finally {
        Pop-Location
    }
}
foreach ($asset in @($configPath, $modelPath, $offlineScript)) {
    if (-not (Test-Path -LiteralPath $asset -PathType Leaf)) {
        throw 'A required public release file is missing; restore the tested repository before running.'
    }
}
if ((Get-LocalSHA256 $configPath) -ne $expectedConfigHash) {
    throw 'Frozen release configuration hash changed. Refusing to run a different candidate.'
}
if ((Get-LocalSHA256 $modelPath) -ne $expectedModelHash) {
    throw 'Frozen release model hash changed. Refusing to run different weights.'
}
if (-not (Test-Path -LiteralPath $runtimeManifest -PathType Leaf)) {
    throw 'Frozen runtime manifest is missing; restore the tested public release.'
}
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'Install the pinned environment first using RUNBOOK.md; this script does not install or download anything.'
}

Push-Location -LiteralPath $projectDirectory
try {
    $environmentCheck = @'
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
project = Path.cwd().resolve()
manifest = json.loads(Path('release/frozen-runtime.json').read_text(encoding='utf-8-sig'))
required_sources = {
    'evaluation/possession_metrics.py', 'evaluation/validate_submission.py', 'games.py',
    'plog/__init__.py', 'plog/__main__.py', 'plog/cli.py', 'plog/config.py', 'plog/data.py',
    'plog/decoder.py', 'plog/events.py', 'plog/features.py', 'plog/inference.py',
    'plog/learning.py', 'plog/physics.py', 'plog/submission.py', 'requirements-plog.lock',
    'run_offline.ps1',
}
if (manifest.get('format') != 'plog-frozen-runtime-v1'
        or manifest.get('source_normalization') != 'CRLF to LF only'
        or set(manifest.get('source_sha256', {})) != required_sources):
    raise SystemExit('Frozen runtime source manifest contract changed.')
for relative, expected in manifest['source_sha256'].items():
    source = Path(relative)
    resolved = (project / source).resolve()
    if source.is_absolute() or '..' in source.parts or project not in resolved.parents:
        raise SystemExit('Frozen runtime source path is outside the public project.')
    try:
        canonical = resolved.read_bytes().replace(b'\r\n', b'\n')
    except OSError:
        raise SystemExit('Frozen runtime source is missing.') from None
    if hashlib.sha256(canonical).hexdigest() != expected:
        raise SystemExit('Frozen runtime source hash changed: ' + relative)
if platform.python_version() != '3.12.10':
    raise SystemExit('Expected CPython 3.12.10; install the pinned environment before using licensed data.')
for line in Path('requirements-plog.lock').read_text(encoding='utf-8').splitlines():
    if '==' not in line or line.lstrip().startswith('#'):
        continue
    name, expected = line.strip().split('==', 1)
    if importlib.metadata.version(name) != expected:
        raise SystemExit('Pinned dependency mismatch: ' + name)
print('Pinned Python/dependency versions, frozen artifacts and canonical runtime source hashes verified.')
'@
    $environmentCheck | & $pythonPath -
    if ($LASTEXITCODE -ne 0) { throw 'Release environment preflight failed.' }
    if ($PreflightOnly) {
        Write-Output 'Preflight passed. No competition path was opened and no predictions were generated.'
        return
    }
    if (-not $DataDir) { throw 'Pass -DataDir with the licensed data folder on this offline machine.' }
    try {
        $inputPath = Get-CallerFileSystemPath $DataDir
        $resolvedOutput = if ($OutputDirectory) { Get-CallerFileSystemPath $OutputDirectory }
                          else { Join-Path $inputPath 'plog-output' }
    } catch {
        throw 'DataDir and OutputDirectory must be valid filesystem paths.'
    }
    $privatePathCheck = @'
from pathlib import Path
import sys
try:
    project = Path(sys.argv[1]).resolve(strict=False)
    candidate = Path(sys.argv[2]).resolve(strict=False)
    allowed = candidate != project and project not in candidate.parents
except (OSError, RuntimeError, ValueError):
    allowed = False
print('true' if allowed else 'false')
'@
    # Resolve Windows junctions, symlinks and existing ancestors of missing paths. Only a
    # boolean crosses this check; no data file is opened or returned by the path guard.
    $privateData = $privatePathCheck | & $pythonPath - $projectDirectory $inputPath
    if ($LASTEXITCODE -ne 0 -or ($privateData -join '').Trim() -ne 'true') {
        throw 'Keep licensed data outside the public Git checkout, including junction targets.'
    }
    if (-not (Test-Path -LiteralPath $inputPath -PathType Container)) {
        throw 'DataDir must be an existing local directory.'
    }
    $resolvedData = (Resolve-Path -LiteralPath $inputPath).ProviderPath
    $privateOutput = $privatePathCheck | & $pythonPath - $projectDirectory $resolvedOutput
    if ($LASTEXITCODE -ne 0 -or ($privateOutput -join '').Trim() -ne 'true') {
        throw 'Keep generated competition files outside the public Git checkout, including junction targets.'
    }
    if (Test-Path -LiteralPath $resolvedOutput -PathType Leaf) {
        throw 'OutputDirectory names an existing file. Choose a new output directory.'
    }
    $outputPath = Join-Path $resolvedOutput 'plog.csv'
    foreach ($existing in @($outputPath, ($outputPath + '.manifest.json'),
                             ($outputPath + '.regenerated.csv'), ($outputPath + '.regenerated.csv.manifest.json'))) {
        if (Test-Path -LiteralPath $existing) {
            throw 'Output files already exist. Choose a new OutputDirectory to preserve the previous submission bytes.'
        }
    }
    $metadataCheck = @'
import sys
import xml.etree.ElementTree as ET
from games import asset_paths
try:
    ET.parse(asset_paths(data_dir=sys.argv[1], source='tactical')['atd_meta'])
except (ET.ParseError, OSError, ValueError, LookupError):
    raise SystemExit('Tactical ATD metadata is missing or invalid. Restore the original tactical bundle; do not edit tracking.') from None
'@
    # The frozen loader remains unchanged. This early structural check gives the
    # operator a fixed, content-free error before inference for malformed XML.
    $metadataCheck | & $pythonPath - $resolvedData
    if ($LASTEXITCODE -ne 0) { throw 'Tactical metadata preflight failed. No predictions were generated.' }
    New-Item -ItemType Directory -Path $resolvedOutput -Force | Out-Null
    & $offlineScript -DataDir $resolvedData -OutputPath $outputPath -ConfigPath $configPath -ModelPath $modelPath
    if ($LASTEXITCODE -ne 0) { throw 'Local inference, validation or regeneration failed.' }
    Write-Output ('Frozen release complete: ' + $outputPath)
    Write-Output 'Both CSVs passed official validation and matched byte-for-byte. Keep all outputs local.'
} finally {
    Pop-Location
}
