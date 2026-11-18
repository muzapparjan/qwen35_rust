<#
.SYNOPSIS
    Downloads the QWQT weight container into src/weights/.

.DESCRIPTION
    The container is a 494,845,446 byte release asset rather than a tracked
    file, so a fresh clone has to fetch it before the crate can compile. The
    download is verified by length and SHA256 and only then moved into place.

.PARAMETER Tag
    Release to take the container from. Defaults to v1.0.0.

.PARAMETER Token
    A GitHub token with read access to the repository. Defaults to the
    GITHUB_TOKEN environment variable, then GH_TOKEN.

    A public repository needs none: the release asset is served anonymously.
    A private one needs this, because the anonymous request answers 404 and the
    download goes through the API asset endpoint instead.

.PARAMETER Destination
    Where to write the container. Defaults to src/weights/qwen3.5-0.8b-q4.qwqt
    next to this script's parent directory.
#>
[CmdletBinding()]
param(
    [string] $Tag = 'v1.0.0',
    [string] $Token = $(if ($env:GITHUB_TOKEN) { $env:GITHUB_TOKEN } else { $env:GH_TOKEN }),
    [string] $Destination
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ExpectedBytes = 494845446
$ExpectedSha256 = '29773850A7AC53122E1544A24748701B55FA59CA63FED89CAE71B9BAABC7DFB9'

$Owner = 'muzapparjan'
$Repository = 'qwen35_rust'
$AssetName = 'qwen3.5-0.8b-q4.qwqt'
$UserAgent = 'qwen35-rust'

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

# The anonymous URL answers 404 on a private repository. The API asset endpoint
# answers 302 to a signed storage URL, which Invoke-WebRequest follows, and the
# Authorization header survives that hop.
function Get-DownloadRequest([string] $Tag, [string] $Token) {
    if (-not $Token) {
        return @{
            Uri = "https://github.com/$Owner/$Repository/releases/download/$Tag/$AssetName"
            Headers = @{ 'User-Agent' = $UserAgent }
        }
    }

    $releaseUri = "https://api.github.com/repos/$Owner/$Repository/releases/tags/$Tag"
    $release = Invoke-RestMethod -Uri $releaseUri -Headers @{
        Authorization = "Bearer $Token"
        Accept = 'application/vnd.github+json'
        'User-Agent' = $UserAgent
    }
    $asset = $release.assets | Where-Object { $_.name -eq $AssetName } | Select-Object -First 1
    if (-not $asset) {
        throw "release ${Tag} has no asset called $AssetName"
    }
    return @{
        Uri = $asset.url
        Headers = @{
            Authorization = "Bearer $Token"
            Accept = 'application/octet-stream'
            'User-Agent' = $UserAgent
        }
    }
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
$request = Get-DownloadRequest -Tag $Tag -Token $Token
Write-Host "downloading $($request.Uri)"
try {
    Invoke-WebRequest -Uri $request.Uri -OutFile $partial -UseBasicParsing -Headers $request.Headers
}
catch {
    if (Test-Path -LiteralPath $partial) {
        Remove-Item -LiteralPath $partial -Force
    }
    $response = $_.Exception.Response
    if ($response -and [int] $response.StatusCode -eq 404 -and -not $Token) {
        throw "$($request.Uri) answered 404. If this repository is private, set GITHUB_TOKEN (or pass -Token) and run again."
    }
    throw
}

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
