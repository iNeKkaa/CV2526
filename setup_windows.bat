@echo off
IF EXIST .venv\Scripts\activate (
    echo Reusing existing virtual environment: .venv
) ELSE (
    echo Creating virtual environment: .venv
    python -m venv .venv
)

call .venv\Scripts\activate

echo Installing / updating PyTorch CUDA build...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Installing / updating requirements...
pip install -r requirements.txt

echo Checking GPU...
python code\verify_gpu.py
pause
