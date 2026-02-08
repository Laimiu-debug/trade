$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

pyinstaller `
  --noconfirm `
  --onefile `
  --windowed `
  --name TrendPicker `
  --add-data "app\streamlit_app.py;." `
  --collect-all streamlit `
  --collect-all plotly `
  app\run_app.py

Write-Host "Build complete. EXE at dist\TrendPicker.exe"
