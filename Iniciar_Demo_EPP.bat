@echo off
REM ======================================================================
REM  Demo de deteccion de EPP en tiempo real con la camara web.
REM  Doble clic para iniciar. La primera vez crea el entorno de Python
REM  (.venv) e instala las dependencias; las siguientes veces parte directo.
REM  Opcional: indicar otra camara, por ejemplo  Iniciar_Demo_EPP.bat 1
REM  En la ventana de video, presiona q para salir.
REM ======================================================================
setlocal
title Demo EPP - camara web
cd /d "%~dp0"

set "CAMARA=%~1"
if "%CAMARA%"=="" set "CAMARA=0"
set "PY=.venv\Scripts\python.exe"
REM Python de uv dentro del proyecto (no en AppData)
set "UV_PYTHON_INSTALL_DIR=%CD%\.python"

REM Prueba de que el entorno sirve para la demo: YOLO + OpenCV con ventanas
set "PRUEBA=import ultralytics, cv2; cv2.namedWindow('x'); cv2.destroyAllWindows()"

REM --- Si el entorno ya esta listo, lanzar la demo de inmediato ---
if exist "%PY%" (
    "%PY%" -c "%PRUEBA%" >nul 2>&1 && goto :demo
)

echo.
echo  Preparando el entorno de Python (la primera vez puede tardar varios minutos)...
echo.

REM --- 1. Conseguir uv (gestor de Python) ---
set "UV=uv"
where uv >nul 2>&1 && goto :uv_listo
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "UV=%USERPROFILE%\.local\bin\uv.exe"
    goto :uv_listo
)
echo  Instalando uv desde https://astral.sh/uv ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if not exist "%USERPROFILE%\.local\bin\uv.exe" (
    echo  [ERROR] No se pudo instalar uv. Revisa la conexion a internet o instalalo a mano:
    echo          https://docs.astral.sh/uv/getting-started/installation/
    goto :error
)
set "UV=%USERPROFILE%\.local\bin\uv.exe"
:uv_listo

REM --- 2. Python 3.11 y entorno virtual ---
if exist "%PY%" (
    "%PY%" -c "import sys" >nul 2>&1 || rmdir /s /q .venv
)
if not exist "%PY%" (
    "%UV%" python install 3.11 || goto :error
    "%UV%" venv .venv --python 3.11 || goto :error
)

REM Si YOLO ya esta instalado, solo falta reparar OpenCV (paso 5)
"%PY%" -c "import ultralytics" >nul 2>&1 && goto :opencv

REM --- 3. PyTorch: con CUDA si hay GPU NVIDIA, si no la version CPU ---
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo  GPU NVIDIA detectada: se instala PyTorch con CUDA 12.4.
    "%UV%" pip install --python "%PY%" torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 || goto :error
) else (
    echo  Sin GPU NVIDIA: se instala PyTorch para CPU.
    "%UV%" pip install --python "%PY%" torch==2.6.0 torchvision==0.21.0 || goto :error
)

REM --- 4. Resto de dependencias del proyecto ---
"%UV%" pip install --python "%PY%" -r modelo\codigo\requirements.txt || goto :error

:opencv
REM --- 5. OpenCV con ventanas ---
REM FiftyOne instala opencv-python-headless (sin ventanas), que comparte la carpeta cv2
REM con opencv-python y puede pisarlo. Se quita y se reinstala opencv-python.
"%UV%" pip uninstall --python "%PY%" opencv-python-headless >nul 2>&1
"%UV%" pip install --python "%PY%" --no-deps --reinstall-package opencv-python -r modelo\codigo\requirements.txt || goto :error
"%PY%" -c "%PRUEBA%" >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] OpenCV quedo sin soporte de ventanas.
    goto :error
)

:demo
if not exist "modelo\pesos\epp_yolo11n.pt" (
    echo  [ERROR] Falta modelo\pesos\epp_yolo11n.pt. Vuelve a clonar o actualizar el repositorio.
    goto :error
)
echo.
echo  Iniciando la demo con la camara %CAMARA%. Presiona q en la ventana de video para salir.
echo.
"%PY%" modelo\codigo\05_detectar_alertas.py --fuente %CAMARA%
if errorlevel 1 goto :error
endlocal
exit /b 0

:error
echo.
echo  La demo no pudo iniciar. Revisa el mensaje de arriba
echo  (seccion "Problemas frecuentes" de modelo\README.md).
pause
endlocal
exit /b 1
