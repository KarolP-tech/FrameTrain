#!/bin/bash

# Install Dependencies und Neustart

echo "📦 Installiere fehlende Dependencies..."
echo ""

cd website

echo "→ Installiere lucide-react..."
npm install

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Installation fehlgeschlagen!"
    exit 1
fi

echo ""
echo "✅ Dependencies installiert!"
echo ""

cd ..

echo "→ Stoppe Services..."
./stop.sh > /dev/null 2>&1

echo "→ Starte Services neu..."
./start.sh

echo ""
echo "✅ Fertig! Die Website sollte jetzt ohne Fehler laufen."
echo ""
echo "Öffne: http://localhost:5001"
echo ""
