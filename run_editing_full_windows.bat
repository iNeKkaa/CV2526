@echo off
call .venv\Scripts\activate
python code\verify_gpu.py
python code\08_run_batch.py --stages editing_full figures --fps 24 --duration 5 --max_side 768 --start_time 1
pause
