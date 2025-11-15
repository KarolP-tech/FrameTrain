#!/bin/bash

# FrameTrain Quick Start
# Installiert alles und startet das System

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Farben
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  FrameTrain Quick Start${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Dieser Script:"
echo "  1. Installiert Rust (falls nötig)"
echo "  2. Führt Setup aus"
echo "  3. Startet alle Services"
echo ""

read -p "Fortfahren? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Abgebrochen."
    exit 0
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Schritt 1: Rust Installation${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Lade Rust falls bereits installiert
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

if command -v rustc &> /dev/null; then
    echo -e "${GREEN}✓ Rust ist bereits installiert: $(rustc --version)${NC}"
else
    echo "→ Installiere Rust..."
    chmod +x install-rust.sh
    ./install-rust.sh
    
    # Lade Rust Environment
    source "$HOME/.cargo/env"
    
    if command -v rustc &> /dev/null; then
        echo -e "${GREEN}✓ Rust erfolgreich installiert${NC}"
    else
        echo -e "${RED}✗ Rust Installation fehlgeschlagen${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Schritt 2: System Check${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

chmod +x test.sh
./test.sh

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}System Check fehlgeschlagen!${NC}"
    echo "Bitte behebe die Fehler und führe ./quickstart.sh erneut aus."
    exit 1
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Schritt 3: Setup${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

chmod +x setup.sh
./setup.sh

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}Setup fehlgeschlagen!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Schritt 4: Environment konfigurieren${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ ! -f "website/.env.local" ]; then
    echo -e "${YELLOW}⚠️  website/.env.local nicht gefunden${NC}"
    echo ""
    read -p "Möchtest du jetzt .env.local erstellen? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp website/.env.local.example website/.env.local
        echo -e "${GREEN}✓ .env.local erstellt${NC}"
        echo ""
        echo -e "${YELLOW}WICHTIG: Bearbeite website/.env.local und füge deine Credentials ein!${NC}"
        echo ""
        read -p "Drücke Enter wenn fertig..."
    else
        echo -e "${YELLOW}⚠️  Überspringe .env.local Setup${NC}"
        echo "Du kannst es später manuell erstellen:"
        echo "  cp website/.env.local.example website/.env.local"
    fi
else
    echo -e "${GREEN}✓ website/.env.local existiert${NC}"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Schritt 5: Services starten${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

read -p "Services jetzt starten? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    chmod +x start.sh
    ./start.sh
    
    echo ""
    echo "Warte 5 Sekunden..."
    sleep 5
    
    echo ""
    chmod +x status.sh
    ./status.sh
else
    echo ""
    echo "Services nicht gestartet."
    echo "Starten mit: ./start.sh"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  🎉 Quick Start abgeschlossen!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Nützliche Befehle:"
echo "  ./status.sh    - Status anzeigen"
echo "  ./stop.sh      - Services stoppen"
echo "  ./restart.sh   - Neu starten"
echo "  tail -f .pids/*.log  - Logs verfolgen"
echo ""
echo "URLs:"
echo "  Website: http://localhost:3000"
echo ""
echo -e "${GREEN}Viel Erfolg! 🚀${NC}"
echo ""
