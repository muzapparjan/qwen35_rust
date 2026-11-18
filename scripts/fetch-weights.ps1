<#
.SYNOPSIS
    Downloads the QWQT weight container into src/weights/.

.DESCRIPTION
    The container is a 494,845,446 byte release asset rather than a tracked
    file, so a fresh clone has to fetch it before the crate can compile. The
    download is verified by length and SHA256 and only then moved into place.

.PARAMETER Url
    Where to download from. Defaults to the v1.0.0 release asset.

.PARAMETER Destination
    Where to write the container. Defaults to src/weights/qwen3.5-0.8b-q4.qwqt
    next to this script's parent directory.
#>
[CmdletBinding()]
param(
    [string] $Url = 'https://github.com/muzapparjan/qwen35_rust/releases/download/v1.0.0/qwen3.5-0.8b-q4.qwqt',
    [string] $Destination
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ExpectedBytes = 494845446
$ExpectedSha256 = '29773850A7AC53122E1544A24748701B55FA59CA63FED89CAE71B9BAABC7DFB9'

if (-not $Destination) {
    $root = Split-Path -Parent $PSScriptRoot
    $Destination = Join-Path $root 'src\weights\qwen3.5-0.8b-q4.qwqt'
}

function Get-FileBytes([string] $Path) {
    (Get-Item -LiteralPath $Path).Length
}

function Get-FileSha256([string] $Path) {
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

if (Test-Path -LiteralPath $Destination) {
    $existingBytes = Get-FileBytes $Destination
    if ($existingBytes -eq $ExpectedBytes -and (Get-FileSha256 $Destination) -eq $ExpectedSha256) {
        Write-Host "already present and verified: $Destination"
        return
    }
    Write-Warning "the existing file does not match; it will be replaced"
}

$partial = Join-Path ([IO.Path]::GetTempPath()) 'qwen3.5-0.8b-q4.qwqt.partial'
if (Test-Path -LiteralPath $partial) {
    Remove-Item -LiteralPath $partial -Force
}

# The progress bar costs more than the download at this size in Windows
# PowerShell 5.1.
$ProgressPreference = 'SilentlyContinue'
Write-Host "downloading $Url"
Invoke-WebRequest -Uri $Url -OutFile $partial -UseBasicParsing

$bytes = Get-FileBytes $partial
if ($bytes -ne $ExpectedBytes) {
    Remove-Item -LiteralPath $partial -Force
    throw "size mismatch: got $bytes bytes, expected $ExpectedBytes"
}

$sha256 = Get-FileSha256 $partial
if ($sha256 -ne $ExpectedSha256) {
    Remove-Item -LiteralPath $partial -Force
    throw "sha256 mismatch: got $sha256, expected $ExpectedSha256"
}

$directory = Split-Path -Parent $Destination
if (-not (Test-Path -LiteralPath $directory)) {
    New-Item -ItemType Directory -Path $directory | Out-Null
}
Move-Item -LiteralPath $partial -Destination $Destination -Force
Write-Host "wrote $Destination ($bytes bytes)"
