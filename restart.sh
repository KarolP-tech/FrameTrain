#!/bin/bash

# FrameTrain Restart Script
# Stoppt und startet alle Services neu

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Farben
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🔄 FrameTrain wird neu gestartet...${NC}"
echo ""

# Stoppe Services
"$SCRIPT_DIR/stop.sh"

echo ""
echo -e "${GREEN}Warte 2 Sekunden...${NC}"
sleep 2
echo ""

# Starte Services
"$SCRIPT_DIR/start.sh" "$@"
