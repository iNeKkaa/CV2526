@echo off
call .venv\Scripts\activate
python code\08_run_batch.py --stages colour matting_rvm matting_bmv2 figures --fps 24 --duration 5 --max_side 768 --start_time 1
pause
