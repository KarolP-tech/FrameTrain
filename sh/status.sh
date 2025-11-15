#!/bin/bash

# FrameTrain Status Script
# Zeigt Status aller Services

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"

# Farben
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  FrameTrain Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -d "$PID_DIR" ]; then
    echo -e "${YELLOW}⚠️  Keine Services laufen${NC}"
    echo ""
    exit 0
fi

check_service() {
    local name=$1
    local pid_file="$PID_DIR/${name}.pid"
    local log_file="$PID_DIR/${name}.log"
    
    if [ ! -f "$pid_file" ]; then
        echo -e "${RED}✗${NC} $name: ${RED}Nicht gestartet${NC}"
        return 1
    fi
    
    local pid=$(cat "$pid_file")
    
    if ps -p "$pid" > /dev/null 2>&1; then
        local uptime=$(ps -o etime= -p "$pid" | tr -d ' ')
        local mem=$(ps -o rss= -p "$pid" | tr -d ' ')
        mem=$((mem / 1024))
        echo -e "${GREEN}✓${NC} $name: ${GREEN}Läuft${NC} (PID: $pid, Uptime: $uptime, RAM: ${mem}MB)"
        
        # Zeige letzte Log-Zeile wenn vorhanden
        if [ -f "$log_file" ] && [ -s "$log_file" ]; then
            local last_log=$(tail -n 1 "$log_file" 2>/dev/null | cut -c1-80)
            if [ -n "$last_log" ]; then
                echo "    └─ $last_log"
            fi
        fi
        return 0
    else
        echo -e "${RED}✗${NC} $name: ${RED}Gestoppt${NC} (PID $pid nicht gefunden)"
        return 1
    fi
}

# Services
SERVICES=("website" "desktop-app")
RUNNING=0
TOTAL=0

for service in "${SERVICES[@]}"; do
    TOTAL=$((TOTAL + 1))
    if check_service "$service"; then
        RUNNING=$((RUNNING + 1))
    fi
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Services: $RUNNING/$TOTAL laufen"
echo ""

if [ $RUNNING -eq 0 ]; then
    echo "Starten mit: ./start.sh"
elif [ $RUNNING -lt $TOTAL ]; then
    echo "Alle starten mit: ./start.sh"
else
    echo "Stoppen mit: ./stop.sh"
fi

echo ""

# Zeige URLs wenn Services laufen
if [ -f "$PID_DIR/website.pid" ]; then
    pid=$(cat "$PID_DIR/website.pid")
    if ps -p "$pid" > /dev/null 2>&1; then
        # Versuche Port aus Log zu extrahieren
        PORT=$(grep -o "localhost:[0-9]*" "$PID_DIR/website.log" 2>/dev/null | head -1 | cut -d: -f2)
        if [ -z "$PORT" ]; then
            PORT="3000"
        fi
        echo "🌐 Website: http://localhost:$PORT"
    fi
fi

if [ -f "$PID_DIR/desktop-app.pid" ]; then
    pid=$(cat "$PID_DIR/desktop-app.pid")
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "🖥️  Desktop-App: Dev-Modus aktiv"
    fi
fi

echo ""
echo "Logs anzeigen:"
echo "  tail -f .pids/*.log"
echo "  # oder einzeln:"
[ -f "$PID_DIR/website.log" ] && echo "  tail -f .pids/website.log"
[ -f "$PID_DIR/desktop-app.log" ] && echo "  tail -f .pids/desktop-app.log"
echo ""
