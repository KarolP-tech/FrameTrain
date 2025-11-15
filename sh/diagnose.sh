#!/bin/bash

# FrameTrain Diagnose Script
# Zeigt detaillierte Informationen über laufende Prozesse und Ports

echo "🔍 FrameTrain Diagnose"
echo "======================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"

# Farben
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Port-Analyse
echo -e "${BLUE}📡 Port-Analyse${NC}"
echo "----------------"
echo ""

check_port() {
    local port=$1
    local name=$2
    
    if lsof -ti:$port > /dev/null 2>&1; then
        echo -e "${YELLOW}Port $port ($name):${NC} ${RED}BELEGT${NC}"
        lsof -ti:$port | while read pid; do
            local cmd=$(ps -p $pid -o command= 2>/dev/null)
            local user=$(ps -p $pid -o user= 2>/dev/null)
            echo "  └─ PID: $pid"
            echo "     User: $user"
            echo "     Command: $cmd"
        done
        echo ""
    else
        echo -e "${YELLOW}Port $port ($name):${NC} ${GREEN}FREI${NC}"
        echo ""
    fi
}

check_port 5001 "Website"
check_port 5002 "Desktop-App Vite"
check_port 1420 "Tauri Dev Server"
check_port 5000 "macOS ControlCenter"

# PID-Dateien
echo -e "${BLUE}📄 PID-Dateien${NC}"
echo "----------------"
echo ""

if [ -d "$PID_DIR" ]; then
    if [ -z "$(ls -A $PID_DIR/*.pid 2>/dev/null)" ]; then
        echo "Keine PID-Dateien gefunden"
    else
        for pid_file in "$PID_DIR"/*.pid; do
            if [ -f "$pid_file" ]; then
                local name=$(basename "$pid_file" .pid)
                local pid=$(cat "$pid_file")
                
                if ps -p "$pid" > /dev/null 2>&1; then
                    echo -e "${GREEN}✓${NC} $name (PID: $pid) ${GREEN}läuft${NC}"
                    local cmd=$(ps -p $pid -o command= 2>/dev/null)
                    echo "  Command: $cmd"
                else
                    echo -e "${RED}✗${NC} $name (PID: $pid) ${RED}läuft NICHT${NC}"
                    echo "  ${YELLOW}⚠️  Verwaiste PID-Datei!${NC}"
                fi
                echo ""
            fi
        done
    fi
else
    echo "PID-Verzeichnis existiert nicht (.pids/)"
fi

echo ""

# Prozess-Suche
echo -e "${BLUE}🔎 FrameTrain Prozesse${NC}"
echo "------------------------"
echo ""

echo "Next.js Prozesse:"
if pgrep -f "next dev" > /dev/null; then
    pgrep -f "next dev" | while read pid; do
        local cmd=$(ps -p $pid -o command= 2>/dev/null)
        if echo "$cmd" | grep -qi "frametrain"; then
            echo -e "  ${GREEN}✓${NC} PID: $pid - FrameTrain Next.js"
        else
            echo -e "  ${YELLOW}?${NC} PID: $pid - Anderer Next.js"
        fi
        echo "    $cmd"
    done
else
    echo "  Keine Next.js Prozesse gefunden"
fi

echo ""

echo "Vite Prozesse:"
if pgrep -f "vite" > /dev/null; then
    pgrep -f "vite" | while read pid; do
        local cmd=$(ps -p $pid -o command= 2>/dev/null)
        if echo "$cmd" | grep -qi "frametrain\|desktop-app"; then
            echo -e "  ${GREEN}✓${NC} PID: $pid - FrameTrain Vite"
        else
            echo -e "  ${YELLOW}?${NC} PID: $pid - Anderer Vite"
        fi
        echo "    $cmd"
    done
else
    echo "  Keine Vite Prozesse gefunden"
fi

echo ""

echo "Tauri Prozesse:"
if pgrep -f "tauri" > /dev/null; then
    pgrep -f "tauri" | while read pid; do
        local cmd=$(ps -p $pid -o command= 2>/dev/null)
        echo -e "  ${GREEN}✓${NC} PID: $pid"
        echo "    $cmd"
    done
else
    echo "  Keine Tauri Prozesse gefunden"
fi

echo ""
echo ""

# Empfehlungen
echo -e "${BLUE}💡 Empfehlungen${NC}"
echo "----------------"
echo ""

# Prüfe auf Probleme
ISSUES=0

# Verwaiste PID-Dateien
if [ -d "$PID_DIR" ]; then
    for pid_file in "$PID_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if ! ps -p "$pid" > /dev/null 2>&1; then
                ISSUES=$((ISSUES + 1))
                echo -e "${YELLOW}⚠️  Verwaiste PID-Datei gefunden: $(basename $pid_file)${NC}"
                echo "   Lösung: ./stop.sh"
            fi
        fi
    done
fi

# Belegte Ports ohne PID-Datei
for port in 5001 5002; do
    if lsof -ti:$port > /dev/null 2>&1; then
        local has_pid_file=false
        if [ -d "$PID_DIR" ]; then
            for pid_file in "$PID_DIR"/*.pid; do
                if [ -f "$pid_file" ]; then
                    local pid=$(cat "$pid_file")
                    if lsof -ti:$port | grep -q "^$pid$"; then
                        has_pid_file=true
                        break
                    fi
                fi
            done
        fi
        
        if [ "$has_pid_file" = false ]; then
            ISSUES=$((ISSUES + 1))
            echo -e "${YELLOW}⚠️  Port $port belegt, aber keine PID-Datei${NC}"
            echo "   Lösung: ./force-kill.sh"
        fi
    fi
done

if [ $ISSUES -eq 0 ]; then
    echo -e "${GREEN}✓ Keine Probleme gefunden!${NC}"
else
    echo ""
    echo -e "${YELLOW}Gefundene Probleme: $ISSUES${NC}"
    echo ""
    echo "Empfohlene Aktionen:"
    echo "  1. ./stop.sh          # Normal stoppen"
    echo "  2. ./force-kill.sh    # Bei hartnäckigen Prozessen"
    echo "  3. ./start.sh         # Neu starten"
fi

echo ""
