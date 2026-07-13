@echo off
setlocal EnableDelayedExpansion

set "WORK_ROOT=C:\research\neuralangelo_test"
set "DATA_PATH=%WORK_ROOT%\datasets\20260703_nonsmooth_match_ds8_posefix"
set "IMAGE_RAW=%DATA_PATH%\images_raw"
set "COLMAP=C:\tools\COLMAP\COLMAP.bat"
set "PYTHON=C:\Users\s7103\anaconda3\envs\neuralangelo_5080\python.exe"
set "SCENE_TYPE=outdoor"
set "VAL_SHORT_SIZE=300"
set "CONFIG_NAME=20260703_nonsmooth_match_ds8_posefix"

if not exist "%DATA_PATH%\sparse" (
  echo Missing sparse directory: "%DATA_PATH%\sparse"
  exit /b 1
)

set "BEST_SPARSE_DIR="
set /a BEST_SPARSE_SIZE=-1

for /d %%D in ("%DATA_PATH%\sparse\*") do (
  if exist "%%~fD\images.bin" (
    set /a MODEL_SIZE=0
    for %%F in ("%%~fD\*.bin") do (
      set /a MODEL_SIZE+=%%~zF
    )
    if !MODEL_SIZE! GTR !BEST_SPARSE_SIZE! (
      set "BEST_SPARSE_DIR=%%~fD"
      set /a BEST_SPARSE_SIZE=!MODEL_SIZE!
    )
  )
)

if not defined BEST_SPARSE_DIR (
  echo No valid sparse model was found under "%DATA_PATH%\sparse".
  exit /b 1
)

echo Using sparse model "!BEST_SPARSE_DIR!" with total .bin size !BEST_SPARSE_SIZE! bytes.

copy /Y "!BEST_SPARSE_DIR!\*.bin" "%DATA_PATH%\sparse\"
if errorlevel 1 exit /b 1
if exist "!BEST_SPARSE_DIR!\project.ini" copy /Y "!BEST_SPARSE_DIR!\project.ini" "%DATA_PATH%\sparse\"

call "%COLMAP%" image_undistorter ^
  --image_path "%IMAGE_RAW%" ^
  --input_path "%DATA_PATH%\sparse" ^
  --output_path "%DATA_PATH%" ^
  --output_type COLMAP
if errorlevel 1 exit /b 1

"%PYTHON%" "%WORK_ROOT%\projects\neuralangelo\scripts\convert_data_to_json.py" ^
  --data_dir "%DATA_PATH%" ^
  --scene_type %SCENE_TYPE%
if errorlevel 1 exit /b 1

"%PYTHON%" "%WORK_ROOT%\projects\neuralangelo\scripts\generate_config.py" ^
  --sequence_name %CONFIG_NAME% ^
  --data_dir "%DATA_PATH%" ^
  --scene_type %SCENE_TYPE% ^
  --val_short_size %VAL_SHORT_SIZE% ^
  --auto_exposure_wb
if errorlevel 1 exit /b 1

echo Done.
echo Dataset: %DATA_PATH%
echo Config: %WORK_ROOT%\projects\neuralangelo\configs\custom\%CONFIG_NAME%.yaml
