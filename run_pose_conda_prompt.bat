@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d C:\neuralangelo

where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  call conda activate neuralangelo_5080
)

set "SETTINGS=C:\neuralangelo\conda_prompt_pose_settings.txt"
set "CHECK_ONLY=0"
set "FORCE=0"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="-CheckOnly" (
  set "CHECK_ONLY=1"
  shift
  goto parse_args
)
if /I "%~1"=="-Force" (
  set "FORCE=1"
  shift
  goto parse_args
)
if /I "%~1"=="-Settings" (
  set "SETTINGS=%~2"
  shift
  shift
  goto parse_args
)
echo Unknown argument: %~1
echo Allowed: -CheckOnly  -Force  -Settings path
exit /b 2

:args_done
set "VIDEO="
set "SEQUENCE=auto"
set "DOWNSAMPLE_RATE=10"
set "SCENE_TYPE=outdoor"
set "VAL_SHORT_SIZE=300"

if not exist "%SETTINGS%" (
  echo Missing settings file: "%SETTINGS%"
  exit /b 1
)

for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%SETTINGS%") do (
  set "KEY=%%~A"
  set "VALUE=%%~B"
  if /I "!KEY!"=="VIDEO" set "VIDEO=!VALUE!"
  if /I "!KEY!"=="SEQUENCE" set "SEQUENCE=!VALUE!"
  if /I "!KEY!"=="DOWNSAMPLE_RATE" set "DOWNSAMPLE_RATE=!VALUE!"
  if /I "!KEY!"=="SCENE_TYPE" set "SCENE_TYPE=!VALUE!"
  if /I "!KEY!"=="VAL_SHORT_SIZE" set "VAL_SHORT_SIZE=!VALUE!"
)

if /I "%SEQUENCE%"=="auto" (
  if defined VIDEO (
    for %%F in ("%VIDEO%") do set "SEQUENCE=%%~nF"
  )
)

if not defined VIDEO (
  echo VIDEO is empty. Edit "%SETTINGS%" and specify the exact video path.
  exit /b 1
)

if not exist "%VIDEO%" (
  echo Missing VIDEO: "%VIDEO%"
  exit /b 1
)

if "%SEQUENCE:\=%" NEQ "%SEQUENCE%" (
  echo Invalid SEQUENCE: "%SEQUENCE%"
  exit /b 1
)
if "%SEQUENCE:/=%" NEQ "%SEQUENCE%" (
  echo Invalid SEQUENCE: "%SEQUENCE%"
  exit /b 1
)
if "%SEQUENCE::=%" NEQ "%SEQUENCE%" (
  echo Invalid SEQUENCE: "%SEQUENCE%"
  exit /b 1
)
if "%SEQUENCE:..=%" NEQ "%SEQUENCE%" (
  echo Invalid SEQUENCE: "%SEQUENCE%"
  exit /b 1
)

if /I not "%SCENE_TYPE%"=="outdoor" if /I not "%SCENE_TYPE%"=="indoor" if /I not "%SCENE_TYPE%"=="object" (
  echo Invalid SCENE_TYPE: "%SCENE_TYPE%"
  exit /b 1
)

set "WORK_ROOT=C:\neuralangelo"
set "FFMPEG=%USERPROFILE%\anaconda3\envs\neuralangelo_5080\Library\bin\ffmpeg.exe"
set "FFPROBE=%USERPROFILE%\anaconda3\envs\neuralangelo_5080\Library\bin\ffprobe.exe"
set "COLMAP=%WORK_ROOT%\tools\COLMAP\bin\colmap.exe"
set "PYTHON=%USERPROFILE%\anaconda3\envs\neuralangelo_5080\python.exe"
set "DATASET_NAME=%SEQUENCE%_ds%DOWNSAMPLE_RATE%"
set "DATA_PATH=%WORK_ROOT%\datasets\%DATASET_NAME%"
set "IMAGE_RAW=%DATA_PATH%\images_raw"
set "SPARSE_PATH=%DATA_PATH%\sparse"
set "SPARSE_CANDIDATES=%DATA_PATH%\sparse_candidates"
set "CONFIG_PATH=%WORK_ROOT%\projects\neuralangelo\configs\custom\%DATASET_NAME%.yaml"
set "SELECT_BEST_SPARSE=%WORK_ROOT%\projects\neuralangelo\scripts\select_best_sparse_model.py"
set "CONVERT_DATA=%WORK_ROOT%\projects\neuralangelo\scripts\convert_data_to_json.py"
set "GENERATE_CONFIG=%WORK_ROOT%\projects\neuralangelo\scripts\generate_config.py"
set "DATABASE=%DATA_PATH%\database.db"

call :require_dir WORK_ROOT "%WORK_ROOT%" || exit /b 1
call :require_file VIDEO "%VIDEO%" || exit /b 1
call :require_file FFMPEG "%FFMPEG%" || exit /b 1
call :require_file FFPROBE "%FFPROBE%" || exit /b 1
call :require_file COLMAP "%COLMAP%" || exit /b 1
call :require_file PYTHON "%PYTHON%" || exit /b 1
call :require_file SELECT_BEST_SPARSE "%SELECT_BEST_SPARSE%" || exit /b 1
call :require_file CONVERT_DATA "%CONVERT_DATA%" || exit /b 1
call :require_file GENERATE_CONFIG "%GENERATE_CONFIG%" || exit /b 1

echo Mode: single camera
echo Settings: %SETTINGS%
echo Sequence: %SEQUENCE%
echo Downsample rate: %DOWNSAMPLE_RATE%
echo Video: %VIDEO%
echo Dataset: %DATA_PATH%
echo Config: %CONFIG_PATH%

echo.
echo == Check ffmpeg can read video ==
"%FFMPEG%" -v error -nostdin -i "%VIDEO%" -frames:v 1 -f null NUL
if errorlevel 1 exit /b 1

echo.
echo == Video info ==
"%FFPROBE%" -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration -show_entries format=duration,size -of default=noprint_wrappers=1 "%VIDEO%"
if errorlevel 1 exit /b 1

echo.
echo == Check COLMAP ==
"%COLMAP%" version
if errorlevel 1 exit /b 1

echo.
echo == Check Python helper ==
"%PYTHON%" "%SELECT_BEST_SPARSE%" --help
if errorlevel 1 exit /b 1

echo.
echo == Check convert_data_to_json.py ==
"%PYTHON%" "%CONVERT_DATA%" --help
if errorlevel 1 exit /b 1

echo.
echo == Check generate_config.py ==
"%PYTHON%" "%GENERATE_CONFIG%" --help
if errorlevel 1 exit /b 1

if "%CHECK_ONLY%"=="1" (
  echo.
  echo Check-only mode passed. No dataset files were created.
  exit /b 0
)

if exist "%DATA_PATH%" (
  if not "%FORCE%"=="1" (
    echo Dataset already exists: "%DATA_PATH%"
    echo Re-run with -Force to regenerate it.
    exit /b 1
  )
  rmdir /s /q "%DATA_PATH%"
)

if not exist "%DATA_PATH%" mkdir "%DATA_PATH%"
if not exist "%IMAGE_RAW%" mkdir "%IMAGE_RAW%"
if not exist "%SPARSE_PATH%" mkdir "%SPARSE_PATH%"
if exist "%SPARSE_CANDIDATES%" rmdir /s /q "%SPARSE_CANDIDATES%"
mkdir "%SPARSE_CANDIDATES%"

echo.
echo == Extract frames ==
"%FFMPEG%" -nostdin -i "%VIDEO%" -vf "select=not(mod(n\,%DOWNSAMPLE_RATE%))" -vsync vfr -q:v 2 "%IMAGE_RAW%\%%06d.jpg"
if errorlevel 1 exit /b 1

echo.
echo == COLMAP feature_extractor ==
"%COLMAP%" feature_extractor ^
  --database_path "%DATABASE%" ^
  --image_path "%IMAGE_RAW%" ^
  --ImageReader.camera_model SIMPLE_RADIAL ^
  --ImageReader.single_camera 1 ^
  --FeatureExtraction.use_gpu 1 ^
  --FeatureExtraction.num_threads 32
if errorlevel 1 exit /b 1

echo.
echo == COLMAP sequential_matcher ==
"%COLMAP%" sequential_matcher ^
  --database_path "%DATABASE%" ^
  --FeatureMatching.use_gpu 1
if errorlevel 1 exit /b 1

echo.
echo == COLMAP mapper ==
"%COLMAP%" mapper ^
  --database_path "%DATABASE%" ^
  --image_path "%IMAGE_RAW%" ^
  --output_path "%SPARSE_CANDIDATES%"
if errorlevel 1 exit /b 1

echo.
echo == Select best sparse model ==
set "BEST_SPARSE_DIR="
for /f "usebackq delims=" %%D in (`"%PYTHON%" "%SELECT_BEST_SPARSE%" --sparse_dir "%SPARSE_CANDIDATES%" --verbose`) do set "BEST_SPARSE_DIR=%%D"
if not defined BEST_SPARSE_DIR (
  echo No valid sparse model was found under "%SPARSE_CANDIDATES%".
  exit /b 1
)
echo Using sparse model: "%BEST_SPARSE_DIR%"
copy "%BEST_SPARSE_DIR%\*.bin" "%SPARSE_PATH%\" /Y
if errorlevel 1 exit /b 1

echo.
echo == COLMAP image_undistorter ==
"%COLMAP%" image_undistorter ^
  --image_path "%IMAGE_RAW%" ^
  --input_path "%SPARSE_PATH%" ^
  --output_path "%DATA_PATH%" ^
  --output_type COLMAP
if errorlevel 1 exit /b 1

echo.
echo == Convert data to JSON ==
"%PYTHON%" "%CONVERT_DATA%" ^
  --data_dir "%DATA_PATH%" ^
  --scene_type %SCENE_TYPE%
if errorlevel 1 exit /b 1

echo.
echo == Generate config ==
"%PYTHON%" "%GENERATE_CONFIG%" ^
  --sequence_name %DATASET_NAME% ^
  --data_dir "%DATA_PATH%" ^
  --scene_type %SCENE_TYPE% ^
  --val_short_size %VAL_SHORT_SIZE% ^
  --auto_exposure_wb
if errorlevel 1 exit /b 1

echo.
echo Done.
echo Dataset: %DATA_PATH%
echo Config: %CONFIG_PATH%
exit /b 0

:require_file
if not exist "%~2" (
  echo Missing %~1: "%~2"
  exit /b 1
)
exit /b 0

:require_dir
if not exist "%~2\" (
  echo Missing %~1: "%~2"
  exit /b 1
)
exit /b 0
