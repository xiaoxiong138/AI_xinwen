@echo off
setlocal

cd /d "%~dp0"

if not exist logs mkdir logs

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set "LOG_FILE=%~dp0logs\scheduled_run_%TS%.log"

set "PYTHON_EXE="
if exist "D:\python\python.exe" set "PYTHON_EXE=D:\python\python.exe"
if not defined PYTHON_EXE for %%i in (python.exe) do set "PYTHON_EXE=%%~$PATH:i"

if not defined PYTHON_EXE (
    echo [%DATE% %TIME%] Python executable not found.>> "%LOG_FILE%"
    exit /b 1
)

"%PYTHON_EXE%" "%~dp0scheduler_runner.py"
set "EXIT_CODE=%ERRORLEVEL%"

endlocal
exit /b %EXIT_CODE%
