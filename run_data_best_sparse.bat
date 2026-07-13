@echo off
setlocal
cd /d C:\neuralangelo
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\neuralangelo\run_data_best_sparse.ps1" %*
exit /b %ERRORLEVEL%
