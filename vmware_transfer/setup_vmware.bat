@echo off
echo Setting up TimeSeries Transformer on VMware
echo ==========================================

REM Create virtual environment
python -m venv venv_gpu
call venv_gpu\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

REM Verify CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

echo Setup complete!
pause
