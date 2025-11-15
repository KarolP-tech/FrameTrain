#!/bin/bash

# FrameTrain Start Script
# Startet alle Services: Website, Desktop-App Dev Mode

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Lade Rust wenn verfügbar
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# Farben
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 FrameTrain wird gestartet...${NC}"
echo ""

# PID Datei Verzeichnis
PID_DIR="$SCRIPT_DIR/.pids"
mkdir -p "$PID_DIR"

# Funktion um Services zu starten
start_service() {
    local name=$1
    local dir=$2
    local command=$3
    local pid_file="$PID_DIR/${name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "${YELLOW}⚠️  $name läuft bereits (PID: $pid)${NC}"
            return 0
        fi
    fi
    
    echo -e "${GREEN}→ Starte $name...${NC}"
    cd "$SCRIPT_DIR/$dir"
    
    # Starte im Hintergrund
    eval "$command" > "$PID_DIR/${name}.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$pid_file"
    
    # Warte kurz um sicherzustellen dass der Prozess läuft
    sleep 2
    
    if ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $name gestartet (PID: $pid)${NC}"
        echo "  Log: $PID_DIR/${name}.log"
    else
        echo -e "${RED}✗ $name konnte nicht gestartet werden${NC}"
        cat "$PID_DIR/${name}.log"
        return 1
    fi
    
    cd "$SCRIPT_DIR"
}

# Prüfe welche Services gestartet werden sollen
if [ "$1" = "website" ] || [ "$1" = "all" ] || [ -z "$1" ]; then
    # Prüfe ob .env.local existiert
    if [ ! -f "website/.env.local" ]; then
        echo -e "${RED}✗ website/.env.local nicht gefunden!${NC}"
        echo "  Erstelle die Datei mit: cp website/.env.local.example website/.env.local"
        exit 1
    fi
    
    start_service "website" "website" "npm run dev"
fi

if [ "$1" = "desktop" ] || [ "$1" = "all" ] || [ -z "$1" ]; then
    # Prüfe ob Rust installiert ist
    if ! command -v rustc &> /dev/null; then
        echo -e "${RED}✗ Rust nicht installiert!${NC}"
        echo "  Führe aus: ./install-rust.sh"
        exit 1
    fi
    
    start_service "desktop-app" "desktop-app" "npm run tauri:dev"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Services gestartet!${NC}"
echo ""
echo "Verfügbare Services:"
if [ -f "$PID_DIR/website.pid" ]; then
    # Versuche Port aus Log zu extrahieren
    PORT=$(grep -o "localhost:[0-9]*" "$PID_DIR/website.log" 2>/dev/null | head -1 | cut -d: -f2)
    if [ -z "$PORT" ]; then
        PORT="3000"
    fi
    echo "  • Website: http://localhost:$PORT"
fi
[ -f "$PID_DIR/desktop-app.pid" ] && echo "  • Desktop-App: Läuft im Dev-Modus"
echo ""
echo "Logs anzeigen:"
echo "  tail -f .pids/*.log"
echo ""
echo "Services stoppen:"
echo "  ./stop.sh"
echo ""
