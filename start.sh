#!/usr/bin/env bash

set -e

echo "=========================================="
echo "     SMD Anomaly Analyzer - Start"
echo "=========================================="
echo

cd "$(dirname "$0")"

if [ ! -f "requirements.txt" ]; then
    echo "[BLAD] Nie znaleziono pliku requirements.txt"
    echo "Upewnij sie, ze skrypt start.sh znajduje sie w katalogu glownym projektu."
    exit 1
fi

if [ ! -f "venv/bin/python" ]; then
    echo "[INFO] Nie znaleziono srodowiska virtualnego."
    echo "[INFO] Tworzenie nowego srodowiska venv..."
    python3 -m venv venv
fi

echo "[INFO] Aktywacja srodowiska..."
source venv/bin/activate

if [ ! -f "venv/.deps_installed" ]; then
    echo "[INFO] Instalacja wymaganych bibliotek..."
    pip install -r requirements.txt
    touch venv/.deps_installed
else
    echo "[INFO] Wykryto istniejace srodowisko z zainstalowanymi bibliotekami."
    echo "[INFO] Pomijam instalacje bibliotek."
fi

echo
echo "[INFO] Uruchamianie aplikacji..."
export PYTHONPATH=.
streamlit run src/app/app.py
