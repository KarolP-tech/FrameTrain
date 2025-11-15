#!/bin/bash

# FrameTrain Stop Script
# Stoppt alle laufenden Services sicher und vollständig

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"

# Farben
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🛑 Stoppe FrameTrain Services...${NC}"
echo ""

# Funktion um Prozess-Baum zu beenden
kill_process_tree() {
    local pid=$1
    local sig=${2:-TERM}
    
    # Finde alle Child-Prozesse
    local children=$(pgrep -P "$pid" 2>/dev/null || true)
    
    # Stoppe zuerst alle Children rekursiv
    for child in $children; do
        kill_process_tree "$child" "$sig"
    done
    
    # Dann den Parent
    if ps -p "$pid" > /dev/null 2>&1; then
        kill -"$sig" "$pid" 2>/dev/null || true
    fi
}

# Funktion um Service zu stoppen
stop_service() {
    local name=$1
    local pid_file="$PID_DIR/${name}.pid"
    
    if [ ! -f "$pid_file" ]; then
        echo -e "${YELLOW}⚠️  $name: Keine PID-Datei gefunden${NC}"
        return 0
    fi
    
    local pid=$(cat "$pid_file")
    
    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  $name: Prozess läuft nicht mehr (PID: $pid)${NC}"
        rm -f "$pid_file"
        return 0
    fi
    
    echo -e "${GREEN}→ Stoppe $name (PID: $pid)...${NC}"
    
    # Versuche graceful shutdown (SIGTERM)
    kill_process_tree "$pid" TERM
    
    # Warte bis zu 10 Sekunden
    local count=0
    while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done
    
    # Wenn noch läuft, force kill (SIGKILL)
    if ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${YELLOW}  Erzwinge Beendigung...${NC}"
        kill_process_tree "$pid" KILL
        sleep 1
    fi
    
    # Prüfe ob wirklich beendet
    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $name gestoppt${NC}"
        rm -f "$pid_file"
        return 0
    else
        echo -e "${RED}✗ Konnte $name nicht stoppen (PID: $pid)${NC}"
        return 1
    fi
}

# Funktion um Port zu prüfen und freizugeben
free_port() {
    local port=$1
    local name=$2
    
    echo -e "${GREEN}→ Prüfe $name (Port $port)...${NC}"
    
    if lsof -ti:$port > /dev/null 2>&1; then
        echo -e "${YELLOW}  Port $port ist belegt, beende Prozesse...${NC}"
        
        # Hole alle PIDs auf diesem Port
        local pids=$(lsof -ti:$port)
        
        for pid in $pids; do
            # Zeige welcher Prozess das ist
            local proc_name=$(ps -p $pid -o comm= 2>/dev/null || echo "unknown")
            echo -e "${YELLOW}  Beende $proc_name (PID: $pid)${NC}"
            
            # Beende Prozess-Baum
            kill_process_tree "$pid" TERM
            sleep 1
            
            # Force kill falls nötig
            if ps -p "$pid" > /dev/null 2>&1; then
                kill_process_tree "$pid" KILL
                sleep 1
            fi
        done
        
        # Prüfe nochmal
        if lsof -ti:$port > /dev/null 2>&1; then
            echo -e "${RED}✗ Port $port konnte nicht freigegeben werden${NC}"
            return 1
        else
            echo -e "${GREEN}✓ Port $port freigegeben${NC}"
            return 0
        fi
    else
        echo -e "${GREEN}✓ Port $port ist frei${NC}"
        return 0
    fi
}

# Stoppe Services über PID-Dateien
if [ -d "$PID_DIR" ]; then
    for pid_file in "$PID_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            service_name=$(basename "$pid_file" .pid)
            stop_service "$service_name"
        fi
    done
    echo ""
fi

# Prüfe und befreie alle Ports
echo -e "${YELLOW}🔍 Prüfe und befreie Ports...${NC}"
echo ""

free_port 5001 "Website"
free_port 5002 "Desktop-App (Vite)"
free_port 1420 "Tauri Dev Server"

echo ""

# Cleanup: Beende alle node/npm/next/tauri Prozesse die zu FrameTrain gehören
echo -e "${YELLOW}🧹 Cleanup: Suche nach verwaisten Prozessen...${NC}"
echo ""

# Finde verwaiste Next.js Prozesse
next_pids=$(pgrep -f "next dev" || true)
if [ -n "$next_pids" ]; then
    echo -e "${YELLOW}Gefundene Next.js Prozesse: $next_pids${NC}"
    for pid in $next_pids; do
        if ps -p $pid -o args= | grep -q "frametrain\|FrameTrain"; then
            echo -e "${YELLOW}  Beende Next.js Prozess (PID: $pid)${NC}"
            kill_process_tree "$pid" TERM
            sleep 1
            if ps -p "$pid" > /dev/null 2>&1; then
                kill_process_tree "$pid" KILL
            fi
        fi
    done
fi

# Finde verwaiste Vite Prozesse
vite_pids=$(pgrep -f "vite" || true)
if [ -n "$vite_pids" ]; then
    echo -e "${YELLOW}Gefundene Vite Prozesse: $vite_pids${NC}"
    for pid in $vite_pids; do
        if ps -p $pid -o args= | grep -q "frametrain\|FrameTrain\|desktop-app"; then
            echo -e "${YELLOW}  Beende Vite Prozess (PID: $pid)${NC}"
            kill_process_tree "$pid" TERM
            sleep 1
            if ps -p "$pid" > /dev/null 2>&1; then
                kill_process_tree "$pid" KILL
            fi
        fi
    done
fi

# Finde verwaiste Tauri Prozesse
tauri_pids=$(pgrep -f "tauri" || true)
if [ -n "$tauri_pids" ]; then
    echo -e "${YELLOW}Gefundene Tauri Prozesse: $tauri_pids${NC}"
    for pid in $tauri_pids; do
        if ps -p $pid -o args= | grep -q "frametrain\|FrameTrain\|desktop-app"; then
            echo -e "${YELLOW}  Beende Tauri Prozess (PID: $pid)${NC}"
            kill_process_tree "$pid" TERM
            sleep 1
            if ps -p "$pid" > /dev/null 2>&1; then
                kill_process_tree "$pid" KILL
            fi
        fi
    done
fi

# Cleanup PID-Verzeichnis
if [ -d "$PID_DIR" ]; then
    rm -rf "$PID_DIR"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Alle Services vollständig gestoppt${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Zeige finale Port-Übersicht
echo "Port-Status:"
for port in 5001 5002 1420; do
    if lsof -ti:$port > /dev/null 2>&1; then
        echo -e "  Port $port: ${RED}BELEGT${NC}"
    else
        echo -e "  Port $port: ${GREEN}FREI${NC}"
    fi
done

echo ""
echo "Neu starten mit: ./start.sh"
echo ""
