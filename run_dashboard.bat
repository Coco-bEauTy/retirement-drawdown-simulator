@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Folder: %cd%
if not exist app.py (
  echo [ERROR] app.py not found. Put this file inside the retirement-dashboard folder, next to app.py.
  pause
  exit /b
)
if not exist actuarial_model.py (
  echo [ERROR] actuarial_model.py not found. Use the whole unzipped folder, not app.py alone.
  pause
  exit /b
)
echo.
echo Step 1/2: installing packages (first time may take a few minutes)...
D:\Python\python.exe -m pip install -r requirements.txt --upgrade
echo.
echo Step 2/2: starting dashboard. Your browser will open. Press Ctrl+C here to stop.
D:\Python\python.exe -m streamlit run app.py
echo.
echo The dashboard has stopped. If you see red error text above, take a screenshot.
pause
