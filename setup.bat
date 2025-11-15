@echo off
REM FrameTrain Setup Script für Windows

echo ========================================
echo FrameTrain Setup
echo ========================================
echo.

REM Prüfe Node.js
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js nicht gefunden. Bitte installiere Node.js 18+
    pause
    exit /b 1
)
echo [OK] Node.js installiert

REM Prüfe Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python nicht gefunden. Bitte installiere Python 3.8+
    pause
    exit /b 1
)
echo [OK] Python installiert

REM Prüfe Rust
where rustc >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Rust nicht gefunden. Bitte installiere Rust: https://rustup.rs/
    pause
    exit /b 1
)
echo [OK] Rust installiert

echo.
echo ========================================
echo.

REM Auswahl
echo Welche Komponenten installieren?
echo 1) Alles (Website + Desktop-App + CLI)
echo 2) Nur Website
echo 3) Nur Desktop-App
echo 4) Nur CLI
echo.
set /p CHOICE="Auswahl (1-4): "

if "%CHOICE%"=="1" (
    set INSTALL_WEBSITE=1
    set INSTALL_DESKTOP=1
    set INSTALL_CLI=1
) else if "%CHOICE%"=="2" (
    set INSTALL_WEBSITE=1
    set INSTALL_DESKTOP=0
    set INSTALL_CLI=0
) else if "%CHOICE%"=="3" (
    set INSTALL_WEBSITE=0
    set INSTALL_DESKTOP=1
    set INSTALL_CLI=0
) else if "%CHOICE%"=="4" (
    set INSTALL_WEBSITE=0
    set INSTALL_DESKTOP=0
    set INSTALL_CLI=1
) else (
    echo [ERROR] Ungueltige Auswahl
    pause
    exit /b 1
)

echo.

REM Website Setup
if "%INSTALL_WEBSITE%"=="1" (
    echo ========================================
    echo Website Setup...
    echo ========================================
    echo.
    
    cd website
    
    echo Installiere Dependencies...
    call npm install
    if %ERRORLEVEL% NEQ 0 exit /b 1
    
    if not exist .env.local (
        echo Erstelle .env.local...
        copy .env.local.example .env.local
        echo [WICHTIG] Bearbeite website/.env.local und fuege deine Credentials ein!
        echo.
    )
    
    echo Setup Prisma...
    call npx prisma generate
    if %ERRORLEVEL% NEQ 0 exit /b 1
    
    cd ..
    echo.
)

REM Desktop-App Setup
if "%INSTALL_DESKTOP%"=="1" (
    echo ========================================
    echo Desktop-App Setup...
    echo ========================================
    echo.
    
    cd desktop-app
    
    echo Installiere Dependencies...
    call npm install
    if %ERRORLEVEL% NEQ 0 exit /b 1
    
    cd ml_backend
    
    if not exist venv (
        echo Erstelle virtuelle Umgebung...
        python -m venv venv
    )
    
    echo Installiere Python Dependencies...
    call venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt
    call deactivate
    
    cd ..\..
    echo.
)

REM CLI Setup
if "%INSTALL_CLI%"=="1" (
    echo ========================================
    echo CLI Setup...
    echo ========================================
    echo.
    
    cd cli
    
    echo Installiere CLI...
    pip install -e .
    if %ERRORLEVEL% NEQ 0 exit /b 1
    
    cd ..
    echo.
)

REM Shared Module
if "%INSTALL_WEBSITE%"=="1" (
    echo ========================================
    echo Shared Module Setup...
    echo ========================================
    echo.
    
    cd shared
    
    echo Installiere Dependencies...
    call npm install
    if %ERRORLEVEL% NEQ 0 exit /b 1
    
    echo Build Shared Module...
    call npm run build
    if %ERRORLEVEL% NEQ 0 exit /b 1
    
    cd ..
    echo.
)

echo ========================================
echo.
echo Setup abgeschlossen!
echo.

if "%INSTALL_WEBSITE%"=="1" (
    echo Website starten:
    echo   cd website ^&^& npm run dev
    echo   -^> http://localhost:3000
    echo.
)

if "%INSTALL_DESKTOP%"=="1" (
    echo Desktop-App starten:
    echo   cd desktop-app ^&^& npm run tauri:dev
    echo.
)

if "%INSTALL_CLI%"=="1" (
    echo CLI verwenden:
    echo   frametrain --help
    echo.
)

echo Weitere Infos:
echo   -^> docs/DEVELOPMENT.md
echo   -^> docs/API.md
echo   -^> docs/DEPLOYMENT.md
echo.
echo Viel Erfolg mit FrameTrain!
echo.

pause
