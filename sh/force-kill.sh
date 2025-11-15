#!/bin/bash

# FrameTrain Force Kill Script
# Beendet ALLE FrameTrain-bezogenen Prozesse mit Gewalt
# Verwende dies nur wenn ./stop.sh nicht funktioniert

echo "⚠️  WARNUNG: Force Kill Script"
echo "================================"
echo ""
echo "Dies beendet ALLE FrameTrain-bezogenen Prozesse mit SIGKILL!"
echo ""
read -p "Bist du sicher? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Abgebrochen."
    exit 0
fi

echo ""
echo "🔥 Beende alle Prozesse..."
echo ""

# Funktion um Prozess-Baum zu killen
kill_tree() {
    local pid=$1
    # Finde alle Children
    local children=$(pgrep -P "$pid" 2>/dev/null || true)
    for child in $children; do
        kill_tree "$child"
    done
    # Kill den Prozess
    kill -9 "$pid" 2>/dev/null || true
}

# Ports killen
echo "→ Beende Prozesse auf Ports..."
for port in 5001 5002 1420; do
    pids=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  Port $port: $pids"
        for pid in $pids; do
            kill_tree "$pid"
        done
    fi
done

# Node/NPM Prozesse mit FrameTrain im Pfad
echo "→ Beende Node.js Prozesse..."
ps aux | grep -i "frametrain\|desktop-app" | grep -v grep | awk '{print $2}' | while read pid; do
    if [ -n "$pid" ]; then
        echo "  Node PID: $pid"
        kill_tree "$pid"
    fi
done

# Next.js
echo "→ Beende Next.js Prozesse..."
pkill -9 -f "next dev" 2>/dev/null || true

# Vite
echo "→ Beende Vite Prozesse..."
pkill -9 -f "vite" 2>/dev/null || true

# Tauri
echo "→ Beende Tauri Prozesse..."
pkill -9 -f "tauri" 2>/dev/null || true
pkill -9 -f "frametrain" 2>/dev/null || true

# Cargo/Rust
echo "→ Beende Cargo Prozesse..."
pkill -9 -f "cargo" 2>/dev/null || true

# Cleanup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"
if [ -d "$PID_DIR" ]; then
    rm -rf "$PID_DIR"
fi

sleep 2

echo ""
echo "✓ Alle Prozesse beendet"
echo ""

# Zeige finale Port-Übersicht
echo "Port-Status:"
for port in 5001 5002 1420; do
    if lsof -ti:$port > /dev/null 2>&1; then
        echo "  Port $port: ❌ NOCH BELEGT"
        lsof -ti:$port | xargs ps -p
    else
        echo "  Port $port: ✓ FREI"
    fi
done

echo ""
