param(
    [string]$ProjectRoot = "",
    [string]$OutputDir = "",
    [string]$WorkDir = ""
)

$ErrorActionPreference = "Stop"
$CollectorRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $CollectorRoot
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $ProjectRoot "data\expanded_phospholipids"
}

if ([string]::IsNullOrWhiteSpace($WorkDir)) {
    $WorkDir = Join-Path $ProjectRoot "data\_downloads"
}

Set-Location $CollectorRoot

python scripts\collect_phospholipid_msms.py `
  --out $OutputDir `
  --work $WorkDir
