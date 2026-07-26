@echo off
setlocal EnableExtensions

set "DATASET=C:\neuralangelo\datasets\20260613_kengdong_ds3"
set "IMAGE_PATH=%DATASET%\images_raw"
set "SPARSE_MODEL=%DATASET%\sparse\0"
set "OUTPUT_ROOT=C:\neuralangelo\output_colmap"
set "MAX_IMAGE_SIZE=960"
set "RUN_NAME=kengdong_%MAX_IMAGE_SIZE%"

set "DENSE_DIR=%OUTPUT_ROOT%\dense_%RUN_NAME%"
set "POISSON_MESH=%OUTPUT_ROOT%\meshed-poisson-%RUN_NAME%.ply"
set "DELAUNAY_MESH=%OUTPUT_ROOT%\meshed-delaunay-%RUN_NAME%.ply"
set "TEXTURE_DIR=%OUTPUT_ROOT%\textured_delaunay_%RUN_NAME%"

echo.
echo COLMAP dense reconstruction
echo Dataset: %DATASET%
echo Resolution: %MAX_IMAGE_SIZE% px
echo.

if not exist "%IMAGE_PATH%\" (
    echo [ERROR] Image directory not found:
    echo %IMAGE_PATH%
    goto :failed
)

if not exist "%SPARSE_MODEL%\cameras.bin" (
    echo [ERROR] Sparse model not found:
    echo %SPARSE_MODEL%
    goto :failed
)

if not exist "%OUTPUT_ROOT%\" mkdir "%OUTPUT_ROOT%"
if errorlevel 1 goto :failed

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"

if exist "%DENSE_DIR%\" (
    echo Backing up existing dense workspace...
    move "%DENSE_DIR%" "%DENSE_DIR%_previous_%STAMP%" >nul
    if errorlevel 1 goto :failed
)

if exist "%POISSON_MESH%" (
    echo Backing up existing Poisson mesh...
    move "%POISSON_MESH%" "%POISSON_MESH%.previous_%STAMP%" >nul
    if errorlevel 1 goto :failed
)

if exist "%DELAUNAY_MESH%" (
    echo Backing up existing Delaunay mesh...
    move "%DELAUNAY_MESH%" "%DELAUNAY_MESH%.previous_%STAMP%" >nul
    if errorlevel 1 goto :failed
)

if exist "%TEXTURE_DIR%\" (
    echo Backing up existing textured mesh...
    move "%TEXTURE_DIR%" "%TEXTURE_DIR%_previous_%STAMP%" >nul
    if errorlevel 1 goto :failed
)

if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" colmap
) else (
    call conda activate colmap
)
if errorlevel 1 (
    echo [ERROR] Could not activate the conda environment named colmap.
    goto :failed
)

where colmap >nul 2>&1
if errorlevel 1 (
    echo [ERROR] colmap.exe was not found after activating the environment.
    goto :failed
)

set "CURRENT_STEP=1/6 image_undistorter"
echo.
echo [1/6] Preparing undistorted images...
colmap image_undistorter ^
    --image_path "%IMAGE_PATH%" ^
    --input_path "%SPARSE_MODEL%" ^
    --output_path "%DENSE_DIR%" ^
    --output_type COLMAP ^
    --max_image_size %MAX_IMAGE_SIZE%
if errorlevel 1 goto :failed

set "CURRENT_STEP=2/6 patch_match_stereo"
echo.
echo [2/6] Computing depth and normal maps...
colmap patch_match_stereo ^
    --workspace_path "%DENSE_DIR%" ^
    --workspace_format COLMAP ^
    --PatchMatchStereo.max_image_size %MAX_IMAGE_SIZE% ^
    --PatchMatchStereo.gpu_index 0 ^
    --PatchMatchStereo.geom_consistency true
if errorlevel 1 goto :failed

set "CURRENT_STEP=3/6 stereo_fusion"
echo.
echo [3/6] Fusing the colored point cloud...
colmap stereo_fusion ^
    --workspace_path "%DENSE_DIR%" ^
    --workspace_format COLMAP ^
    --input_type geometric ^
    --output_path "%DENSE_DIR%\fused.ply"
if errorlevel 1 goto :failed

set "CURRENT_STEP=4/6 poisson_mesher"
echo.
echo [4/6] Building the Poisson mesh...
colmap poisson_mesher ^
    --input_path "%DENSE_DIR%\fused.ply" ^
    --output_path "%POISSON_MESH%"
if errorlevel 1 goto :failed

set "CURRENT_STEP=5/6 delaunay_mesher"
echo.
echo [5/6] Building the Delaunay mesh...
echo This is the most RAM-intensive step and can take several hours.
colmap delaunay_mesher ^
    --input_path "%DENSE_DIR%" ^
    --input_type dense ^
    --output_path "%DELAUNAY_MESH%"
if errorlevel 1 goto :failed

set "CURRENT_STEP=6/6 mesh_texturer"
echo.
echo [6/6] Applying image textures to the Delaunay mesh...
colmap mesh_texturer ^
    --workspace_path "%DENSE_DIR%" ^
    --input_path "%DELAUNAY_MESH%" ^
    --output_path "%TEXTURE_DIR%"
if errorlevel 1 goto :failed

echo.
echo [DONE] Reconstruction completed.
echo Point cloud: %DENSE_DIR%\fused.ply
echo Poisson mesh: %POISSON_MESH%
echo Delaunay mesh: %DELAUNAY_MESH%
echo Textured mesh: %TEXTURE_DIR%
echo.
pause
exit /b 0

:failed
echo.
if defined CURRENT_STEP echo [ERROR] Pipeline stopped during %CURRENT_STEP%.
echo Review the COLMAP message shown above.
echo.
pause
exit /b 1
