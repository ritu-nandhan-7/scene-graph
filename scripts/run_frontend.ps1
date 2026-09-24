<#
    Scene Graph Explorer - start the React (Vite) frontend.

    Usage:
        powershell -ExecutionPolicy Bypass -File scripts\run_frontend.ps1

    Serves http://127.0.0.1:5173
    Stop with Ctrl+C.

    Requires the backend (scripts\run_backend.ps1) to be running in a
    second terminal - the frontend calls POST /analyze on port 8000.
#>
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$frontend = Join-Path $repoRoot 'frontend'
Set-Location $frontend

if (-not (Test-Path (Join-Path $frontend 'node_modules'))) {
    Write-Host 'node_modules missing - installing frontend dependencies...'
    npm install
}

Write-Host '============================================='
Write-Host ' Scene Graph Explorer - frontend'
Write-Host '============================================='
Write-Host "Directory  : $frontend"
Write-Host 'Open       : http://127.0.0.1:5173'
Write-Host ''

npm run dev
