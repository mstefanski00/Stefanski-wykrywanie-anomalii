@echo off
title SMD Anomaly Analyzer
color 0A


cd /d "%~dp0"

if not exist "requirements.txt" (
    echo [BLAD] Nie znaleziono pliku requirements.txt
    echo Upewnij sie, ze plik .bat znajduje sie w folderze glownym projektu.
    pause
    exit /b 1
)

if not exist ".\venv\Scripts\python.exe" (
    echo [INFO] Nie znaleziono srodowiska virtualnego.
    echo [INFO] Tworzenie nowego srodowiska venv...
    python -m venv venv
    if errorlevel 1 (
        echo [BLAD] Nie udalo sie utworzyc srodowiska virtualnego.
        echo Sprawdz, czy Python jest zainstalowany i dodany do PATH.
        pause
        exit /b 1
    )

    echo [INFO] Aktywacja srodowiska...
    call .\venv\Scripts\activate.bat

    echo [INFO] Instalacja wymaganych bibliotek...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [BLAD] Nie udalo sie zainstalowac wymaganych bibliotek.
        pause
        exit /b 1
    )
) else (
    echo [INFO] Wykryto istniejace srodowisko venv.
    echo [INFO] Pomijam instalacje bibliotek.
    call .\venv\Scripts\activate.bat
)

echo.
echo [INFO] Uruchamianie aplikacji...
set PYTHONPATH=.
streamlit run src\app\app.py

echo.
echo [INFO] Aplikacja zostala zamknieta.
pause