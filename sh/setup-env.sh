#!/bin/bash

# FrameTrain Environment Setup Helper
# Hilft beim interaktiven Erstellen der .env.local Datei

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/website/.env.local"

# Farben
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

clear
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  FrameTrain Environment Setup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Dieser Script hilft dir beim Erstellen der .env.local Datei."
echo ""

# Prüfe ob Datei existiert
if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}⚠️  .env.local existiert bereits!${NC}"
    echo ""
    read -p "Überschreiben? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Abgebrochen."
        exit 0
    fi
    echo ""
fi

echo -e "${GREEN}Los geht's! Beantworte die folgenden Fragen:${NC}"
echo ""
echo -e "${CYAN}Tipp: Drücke Enter für Standardwerte${NC}"
echo ""

# ============================================
# DATABASE
# ============================================
echo -e "${BLUE}1. Datenbank${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Wähle deine Datenbank-Option:"
echo "  1) PostgreSQL lokal (localhost)"
echo "  2) Supabase (Cloud)"
echo "  3) Railway (Cloud)"
echo "  4) Eigene Connection String"
echo ""
read -p "Auswahl (1-4) [1]: " db_choice
db_choice=${db_choice:-1}

case $db_choice in
    1)
        DATABASE_URL="postgresql://localhost:5432/frametrain"
        echo -e "${GREEN}✓ Lokal: $DATABASE_URL${NC}"
        ;;
    2)
        echo ""
        echo "Supabase Connection String (von Project Settings → Database):"
        read -p "> " DATABASE_URL
        ;;
    3)
        echo ""
        echo "Railway Connection String:"
        read -p "> " DATABASE_URL
        ;;
    4)
        echo ""
        echo "Eigene Connection String:"
        read -p "> " DATABASE_URL
        ;;
    *)
        DATABASE_URL="postgresql://localhost:5432/frametrain"
        ;;
esac

echo ""

# ============================================
# JWT SECRET
# ============================================
echo -e "${BLUE}2. JWT Secret${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Generiere einen zufälligen JWT Secret?"
echo "  1) Ja, automatisch generieren (empfohlen)"
echo "  2) Nein, ich gebe selbst einen ein"
echo ""
read -p "Auswahl (1-2) [1]: " jwt_choice
jwt_choice=${jwt_choice:-1}

if [ "$jwt_choice" = "1" ]; then
    JWT_SECRET=$(openssl rand -base64 32)
    echo -e "${GREEN}✓ Generiert: $JWT_SECRET${NC}"
else
    echo ""
    echo "JWT Secret eingeben (min. 32 Zeichen):"
    read -p "> " JWT_SECRET
fi

echo ""

# ============================================
# STRIPE
# ============================================
echo -e "${BLUE}3. Stripe Keys${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${CYAN}Hol deine Keys von: https://dashboard.stripe.com/test/apikeys${NC}"
echo ""

echo "Stripe Secret Key (sk_test_...):"
read -p "> " STRIPE_SECRET_KEY

echo ""
echo "Stripe Publishable Key (pk_test_...):"
read -p "> " STRIPE_PUBLISHABLE_KEY

echo ""
echo -e "${CYAN}Webhook Secret von: https://dashboard.stripe.com/test/webhooks${NC}"
echo "Stripe Webhook Secret (whsec_...):"
read -p "> " STRIPE_WEBHOOK_SECRET

echo ""

# ============================================
# API URLs
# ============================================
echo -e "${BLUE}4. API URLs${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

read -p "API URL [http://localhost:3000]: " API_URL
API_URL=${API_URL:-http://localhost:3000}

NEXT_PUBLIC_API_URL=$API_URL

read -p "App Download Base URL [http://localhost:3000/downloads]: " APP_DOWNLOAD_BASE_URL
APP_DOWNLOAD_BASE_URL=${APP_DOWNLOAD_BASE_URL:-http://localhost:3000/downloads}

echo ""

# ============================================
# Zusammenfassung
# ============================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Zusammenfassung${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "DATABASE_URL: $DATABASE_URL"
echo "JWT_SECRET: ${JWT_SECRET:0:20}..."
echo "STRIPE_SECRET_KEY: ${STRIPE_SECRET_KEY:0:20}..."
echo "STRIPE_PUBLISHABLE_KEY: ${STRIPE_PUBLISHABLE_KEY:0:20}..."
echo "STRIPE_WEBHOOK_SECRET: ${STRIPE_WEBHOOK_SECRET:0:20}..."
echo "API_URL: $API_URL"
echo ""

read -p "Alles korrekt? Datei erstellen? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Abgebrochen. Führe ./setup-env.sh erneut aus."
    exit 0
fi

# ============================================
# Datei erstellen
# ============================================
echo ""
echo -e "${GREEN}→ Erstelle .env.local...${NC}"

cat > "$ENV_FILE" << EOF
# FrameTrain Environment Configuration
# Generiert am: $(date)

# ============================================
# DATABASE
# ============================================
DATABASE_URL="$DATABASE_URL"

# ============================================
# JWT SECRET (Authentication)
# ============================================
JWT_SECRET="$JWT_SECRET"

# ============================================
# STRIPE (Payment)
# ============================================
STRIPE_SECRET_KEY="$STRIPE_SECRET_KEY"
STRIPE_PUBLISHABLE_KEY="$STRIPE_PUBLISHABLE_KEY"
STRIPE_WEBHOOK_SECRET="$STRIPE_WEBHOOK_SECRET"

# ============================================
# API CONFIGURATION
# ============================================
API_URL="$API_URL"
NEXT_PUBLIC_API_URL="$NEXT_PUBLIC_API_URL"

# ============================================
# APP DOWNLOAD
# ============================================
APP_DOWNLOAD_BASE_URL="$APP_DOWNLOAD_BASE_URL"

# ============================================
# ENVIRONMENT
# ============================================
NODE_ENV="development"
EOF

echo -e "${GREEN}✓ .env.local erstellt!${NC}"
echo ""

# ============================================
# Next Steps
# ============================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Nächste Schritte${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ "$db_choice" = "1" ]; then
    echo -e "${YELLOW}⚠️  PostgreSQL lokal gewählt${NC}"
    echo ""
    echo "Stelle sicher dass PostgreSQL läuft:"
    echo "  brew services start postgresql@15"
    echo ""
    echo "Erstelle die Datenbank:"
    echo "  createdb frametrain"
    echo ""
fi

echo "1. Datenbank initialisieren:"
echo "   cd website"
echo "   npx prisma generate"
echo "   npx prisma db push"
echo ""

echo "2. Services starten:"
echo "   ./start.sh"
echo ""

echo "3. Status prüfen:"
echo "   ./status.sh"
echo ""

echo -e "${GREEN}🎉 Setup abgeschlossen!${NC}"
echo ""
echo "Hilfe: ./PAYMENT_SETUP.md"
echo ""
