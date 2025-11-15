#!/bin/bash

# FrameTrain Website Database Init
# Initialisiert die Datenbank korrekt mit allen Environment-Variables

cd "$(dirname "$0")"

echo "🗄️  Initialisiere Datenbank..."
echo ""

# Prüfe ob .env.local existiert
if [ ! -f ".env.local" ]; then
    echo "❌ .env.local nicht gefunden!"
    echo ""
    echo "Erstelle .env.local aus dem Template..."
    if [ -f ".env.local.example" ]; then
        cp .env.local.example .env.local
        echo "✓ .env.local erstellt"
        echo ""
        echo "⚠️  WICHTIG: Bearbeite .env.local und füge deine Credentials ein!"
    else
        echo "❌ Auch .env.local.example nicht gefunden!"
        exit 1
    fi
fi

# Lade Environment Variables
export $(cat .env.local | grep -v '^#' | xargs)

# Prüfe ob DATABASE_URL gesetzt ist
if [ -z "$DATABASE_URL" ]; then
    echo "❌ DATABASE_URL ist nicht gesetzt in .env.local"
    echo ""
    echo "Füge folgende Zeile zu .env.local hinzu:"
    echo 'DATABASE_URL="file:./dev.db"'
    exit 1
fi

echo "✓ DATABASE_URL gefunden: $DATABASE_URL"
echo ""

# Generiere Prisma Client
echo "→ Generiere Prisma Client..."
npx prisma generate

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Fehler beim Generieren des Prisma Clients"
    exit 1
fi

echo "✓ Prisma Client generiert"
echo ""

# Pushe Schema zur Datenbank
echo "→ Erstelle/Aktualisiere Datenbank-Schema..."
npx prisma db push --skip-generate

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Fehler beim Pushen des Schemas"
    exit 1
fi

echo "✓ Datenbank-Schema angewendet"
echo ""

# Prüfe ob dev.db erstellt wurde
if [ -f "prisma/dev.db" ]; then
    echo "✓ Datenbank-Datei erstellt: prisma/dev.db"
else
    echo "⚠️  Warnung: prisma/dev.db wurde nicht gefunden"
fi

echo ""
echo "========================================="
echo "✓ Datenbank erfolgreich initialisiert!"
echo "========================================="
echo ""
echo "Optional: Öffne Prisma Studio zur Verwaltung:"
echo "  npx prisma studio"
echo ""
