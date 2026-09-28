$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

& .\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
& .\.venv\Scripts\python.exe -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --name recallx-engine `
    --paths . `
    --collect-all sentence_transformers `
    --collect-all onnxruntime `
    --collect-all openvino `
    --collect-all optimum `
    --collect-all optimum_intel `
    --hidden-import app.main `
    run_engine.py
