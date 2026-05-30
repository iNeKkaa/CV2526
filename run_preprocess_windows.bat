@echo off
call .venv\Scripts\activate
python code\00_prepare_videos.py --fps 24 --duration 5 --max_side 768 --start_time 1
pause
