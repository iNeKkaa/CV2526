@echo off
call .venv\Scripts\activate
python code\08_run_batch.py --stages colour_backup matting_light figures --fps 24 --duration 5 --max_side 768 --start_time 1
pause
