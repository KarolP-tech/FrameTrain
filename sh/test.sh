#!/bin/bash

# FrameTrain Test & Verification Script
# Prüft ob alle Komponenten korrekt funktionieren

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
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  FrameTrain Verification & Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

ERRORS=0
WARNINGS=0
SUCCESS=0

# Test Funktion
test_item() {
    local name="$1"
    local command="$2"
    local critical="${3:-false}"
    
    echo -n "Testing $name... "
    
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        SUCCESS=$((SUCCESS + 1))
        return 0
    else
        if [ "$critical" = "true" ]; then
            echo -e "${RED}✗ FEHLER${NC}"
            ERRORS=$((ERRORS + 1))
        else
            echo -e "${YELLOW}⚠ WARNUNG${NC}"
            WARNINGS=$((WARNINGS + 1))
        fi
        return 1
    fi
}

echo -e "${BLUE}1. System Requirements${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_item "Node.js installiert" "command -v node" true
test_item "npm installiert" "command -v npm" true
test_item "Python3 installiert" "command -v python3" true
test_item "pip installiert" "command -v pip3" true
test_item "Rust installiert" "command -v rustc" true
test_item "Cargo installiert" "command -v cargo" true
test_item "PostgreSQL installiert" "command -v psql" false
echo ""

echo -e "${BLUE}2. Project Structure${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_item "website/ existiert" "[ -d website ]" true
test_item "desktop-app/ existiert" "[ -d desktop-app ]" true
test_item "cli/ existiert" "[ -d cli ]" true
test_item "shared/ existiert" "[ -d shared ]" true
test_item "docs/ existiert" "[ -d docs ]" true
echo ""

echo -e "${BLUE}3. Website${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_item "package.json existiert" "[ -f website/package.json ]" true
test_item "node_modules existiert" "[ -d website/node_modules ]" false
test_item ".env.local.example existiert" "[ -f website/.env.local.example ]" true
test_item "Prisma Schema existiert" "[ -f website/prisma/schema.prisma ]" true
test_item "API Routes existieren" "[ -d website/src/app/api ]" true
test_item "Components existieren" "[ -d website/src/components ]" true
echo ""

echo -e "${BLUE}4. Desktop App${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_item "package.json existiert" "[ -f desktop-app/package.json ]" true
test_item "Tauri config existiert" "[ -f desktop-app/src-tauri/tauri.conf.json ]" true
test_item "Cargo.toml existiert" "[ -f desktop-app/src-tauri/Cargo.toml ]" true
test_item "SQLite Schema existiert" "[ -f desktop-app/schema.sql ]" true
test_item "ML Backend existiert" "[ -d desktop-app/ml_backend ]" true
test_item "Python requirements.txt existiert" "[ -f desktop-app/ml_backend/requirements.txt ]" true
test_item "Rust src/main.rs existiert" "[ -f desktop-app/src-tauri/src/main.rs ]" true
test_item "Rust database.rs existiert" "[ -f desktop-app/src-tauri/src/database.rs ]" true
echo ""

echo -e "${BLUE}5. CLI${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_item "pyproject.toml existiert" "[ -f cli/pyproject.toml ]" true
test_item "CLI Module existiert" "[ -d cli/frametrain ]" true
test_item "Commands existieren" "[ -d cli/frametrain/commands ]" true
test_item "CLI installiert" "command -v frametrain" false
echo ""

echo -e "${BLUE}6. Scripts${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_item "setup.sh existiert" "[ -f setup.sh ]" true
test_item "start.sh existiert" "[ -f start.sh ]" true
test_item "stop.sh existiert" "[ -f stop.sh ]" true
test_item "restart.sh existiert" "[ -f restart.sh ]" true
test_item "status.sh existiert" "[ -f status.sh ]" true
test_item "install-rust.sh existiert" "[ -f install-rust.sh ]" true
test_item "setup.sh ausführbar" "[ -x setup.sh ]" true
test_item "start.sh ausführbar" "[ -x start.sh ]" true
test_item "stop.sh ausführbar" "[ -x stop.sh ]" true
echo ""

echo -e "${BLUE}7. Documentation${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_item "README.md existiert" "[ -f README.md ]" true
test_item "DEVELOPMENT.md existiert" "[ -f docs/DEVELOPMENT.md ]" true
test_item "DEPLOYMENT.md existiert" "[ -f docs/DEPLOYMENT.md ]" true
test_item "API.md existiert" "[ -f docs/API.md ]" true
test_item "SCRIPTS.md existiert" "[ -f docs/SCRIPTS.md ]" true
test_item "PROJECT_STATUS.md existiert" "[ -f PROJECT_STATUS.md ]" true
echo ""

echo -e "${BLUE}8. Configuration Files${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
test_item "website/.gitignore" "[ -f website/.gitignore ] || [ -f .gitignore ]" true
test_item "website/tsconfig.json" "[ -f website/tsconfig.json ]" true
test_item "website/tailwind.config.js" "[ -f website/tailwind.config.js ]" true
test_item "website/next.config.js" "[ -f website/next.config.js ]" true
test_item "desktop-app/tsconfig.json" "[ -f desktop-app/tsconfig.json ]" true
test_item "desktop-app/tailwind.config.js" "[ -f desktop-app/tailwind.config.js ]" true
echo ""

echo -e "${BLUE}9. Syntax Checks${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "website/package.json" ]; then
    test_item "website/package.json valid" "node -e 'JSON.parse(require(\"fs\").readFileSync(\"website/package.json\"))'" true
fi

if [ -f "desktop-app/package.json" ]; then
    test_item "desktop-app/package.json valid" "node -e 'JSON.parse(require(\"fs\").readFileSync(\"desktop-app/package.json\"))'" true
fi

if [ -f "website/prisma/schema.prisma" ]; then
    test_item "Prisma Schema syntax" "cd website && npx prisma validate" false
fi

echo ""

# Zusammenfassung
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Test Results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}✓ Erfolg: $SUCCESS${NC}"
echo -e "${YELLOW}⚠ Warnungen: $WARNINGS${NC}"
echo -e "${RED}✗ Fehler: $ERRORS${NC}"
echo ""

if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}  KRITISCHE FEHLER GEFUNDEN!${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Bitte behebe die Fehler und führe ./test.sh erneut aus."
    echo ""
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  Einige Warnungen gefunden${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Das System ist funktionsfähig, aber einige optionale"
    echo "Komponenten fehlen oder sind nicht konfiguriert."
    echo ""
    echo "Nächste Schritte:"
    echo "  1. Fehlende Dependencies installieren"
    echo "  2. Services mit ./start.sh starten"
    echo ""
else
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  🎉 ALLE TESTS ERFOLGREICH!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "FrameTrain ist bereit!"
    echo ""
    echo "Nächste Schritte:"
    echo "  1. Konfiguriere .env.local:"
    echo "     cp website/.env.local.example website/.env.local"
    echo ""
    echo "  2. Starte Services:"
    echo "     ./start.sh"
    echo ""
    echo "  3. Prüfe Status:"
    echo "     ./status.sh"
    echo ""
fi
