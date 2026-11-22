@echo off
REM Starts the demo website on this computer (no internet needed) and opens it in the browser.
REM Keep this window open while you use the demo; close it to stop.
cd /d "%~dp0"
where python >nul 2>nul && (python serve.py & goto :eof)
where py >nul 2>nul && (py serve.py & goto :eof)
if exist "C:\APPS\Anaconda3\python.exe" ("C:\APPS\Anaconda3\python.exe" serve.py & goto :eof)
echo Python was not found. Install Python, or use the online version of the demo.
pause
