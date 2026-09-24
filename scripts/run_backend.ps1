<#
    Scene Graph Explorer - start the FastAPI backend.

    Usage:
        powershell -ExecutionPolicy Bypass -File scripts\run_backend.ps1

    Serves http://127.0.0.1:8000  (Swagger UI at /docs)
    Stop with Ctrl+C.

    The first start loads YOLO-World, CLIP and the relationship checkpoint
    ONCE.  On this CPU-only machine that takes roughly 1-2 minutes; every
    later /analyze request reuses the already-loaded pipeline.
#>
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# Prefer the project virtual environment when it exists.
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
$python = if (Test-Path $venvPython) { $venvPython } else { 'python' }

Write-Host '============================================='
Write-Host ' Scene Graph Explorer - backend'
Write-Host '============================================='
Write-Host "Repository : $repoRoot"
Write-Host "Python     : $python"
Write-Host 'Health     : http://127.0.0.1:8000/health'
Write-Host 'Swagger    : http://127.0.0.1:8000/docs'
Write-Host ''
Write-Host 'Loading models on startup - this takes 1-2 minutes on CPU...'
Write-Host ''

& $python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
