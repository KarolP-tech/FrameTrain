#!/bin/bash

# Quick Fix für FrameTrain 500 Errors
# Behebt häufige Probleme mit der Datenbank und API

cd "$(dirname "$0")"

echo "🔧 FrameTrain Quick Fix"
echo "======================="
echo ""

# Farben
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Dieser Script behebt:"
echo "  • 500 Internal Server Errors"
echo "  • Datenbank-Probleme"
echo "  • Prisma Client Probleme"
echo "  • SQLite JSON-Feld Probleme"
echo ""

read -p "Fortfahren? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

echo ""
echo "→ Stoppe laufende Services..."
./stop.sh > /dev/null 2>&1

echo "→ Wechsle ins Website-Verzeichnis..."
cd website

echo "→ Prüfe .env.local..."
if [ ! -f ".env.local" ]; then
    echo -e "${YELLOW}  .env.local nicht gefunden, erstelle aus Template...${NC}"
    if [ -f ".env.local.example" ]; then
        cp .env.local.example .env.local
        echo -e "${GREEN}  ✓ .env.local erstellt${NC}"
    fi
fi

echo "→ Lösche alte Datenbank (falls vorhanden)..."
rm -f prisma/dev.db prisma/dev.db-journal

echo "→ Führe Datenbank-Initialisierung aus..."
chmod +x init-db.sh
./init-db.sh

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Quick Fix erfolgreich abgeschlossen!${NC}"
    echo ""
    echo "Starte jetzt die Services neu:"
    echo "  cd .. && ./start.sh"
    echo ""
    echo "Oder nur die Website:"
    echo "  npm run dev"
    echo ""
else
    echo ""
    echo -e "${RED}✗ Quick Fix fehlgeschlagen!${NC}"
    echo ""
    echo "Manuelle Schritte:"
    echo "  1. Prüfe .env.local"
    echo "  2. Stelle sicher dass DATABASE_URL=\"file:./dev.db\" gesetzt ist"
    echo "  3. Führe aus: cd website && ./init-db.sh"
    echo ""
    exit 1
fi
