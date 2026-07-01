@echo off
setlocal enabledelayedexpansion

echo === Building and Publishing planetbridging to PyPI ===
echo.

cd /d "%~dp0"

call conda activate base 2>nul
if errorlevel 1 call "%USERPROFILE%\miniconda3\Scripts\activate.bat" base 2>nul

for /f "tokens=2 delims==" %%a in ('findstr /r "^name = " pyproject.toml') do set PKG_NAME=%%~a
for /f "tokens=2 delims==" %%a in ('findstr /r "^version = " pyproject.toml') do set PKG_VERSION=%%~a
set PKG_NAME=%PKG_NAME:"=%
set PKG_VERSION=%PKG_VERSION:"=%

echo Package: %PKG_NAME% %PKG_VERSION%
echo Note:   loom-stream Go binary is NOT included in the PyPI wheel.
echo.

echo Cleaning previous builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
for /d %%i in (*.egg-info src\*.egg-info) do rmdir /s /q "%%i"

echo Building package...
python -m build
if errorlevel 1 (
    echo.
    echo [!] Build failed. Install packaging tools:
    echo     pip install build twine
    exit /b 1
)

echo.
echo Build complete!
echo.

python -m twine check dist\*
if errorlevel 1 (
    echo.
    echo [!] Twine check failed. Install with:
    echo     pip install build twine
    exit /b 1
)

echo Package passes twine checks
echo.
echo Files to upload:
dir /b dist\
echo.

set /p CONFIRM="Upload %PKG_NAME% %PKG_VERSION% to PyPI? (y/N): "
if /i "%CONFIRM%"=="y" (
    echo Uploading to PyPI...
    python -m twine upload dist\*
    if errorlevel 1 (
        echo.
        echo [!] Upload failed. Check PyPI credentials in %%USERPROFILE%%\.pypirc
        exit /b 1
    )
    echo.
    echo === Published Successfully ===
    echo View at: https://pypi.org/project/%PKG_NAME%/
    echo.
    echo Install with: pip install %PKG_NAME%
    echo Optional:    pip install %PKG_NAME%[pytorch]
) else (
    echo Upload cancelled.
    echo.
    echo To upload manually:
    echo     python -m twine upload dist\*
)
