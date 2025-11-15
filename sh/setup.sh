#!/bin/bash

# FrameTrain Setup Script
# Automatisiert das Setup aller Projekt-Komponenten

set -e

echo "🚀 FrameTrain Setup"
echo "===================="
echo ""

# Farben
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Funktion für Erfolg-Meldungen
success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Funktion für Info-Meldungen
info() {
    echo -e "${YELLOW}→${NC} $1"
}

# Funktion für Fehler-Meldungen
error() {
    echo -e "${RED}✗${NC} $1"
}

# Prüfe Voraussetzungen
echo "📋 Prüfe Voraussetzungen..."
echo ""

# Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v)
    success "Node.js $NODE_VERSION installiert"
else
    error "Node.js nicht gefunden. Bitte installiere Node.js 18+"
    exit 1
fi

# Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    success "$PYTHON_VERSION installiert"
else
    error "Python3 nicht gefunden. Bitte installiere Python 3.8+"
    exit 1
fi

# Rust
if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version)
    success "$RUST_VERSION installiert"
else
    # Versuche Rust aus .cargo/env zu laden
    if [ -f "$HOME/.cargo/env" ]; then
        info "Lade Rust aus ~/.cargo/env..."
        source "$HOME/.cargo/env"
        if command -v rustc &> /dev/null; then
            RUST_VERSION=$(rustc --version)
            success "$RUST_VERSION installiert"
        else
            error "Rust nicht gefunden. Bitte installiere Rust: https://rustup.rs/"
            echo "  Oder führe aus: ./install-rust.sh"
            echo "  Dann: source ~/.cargo/env"
            exit 1
        fi
    else
        error "Rust nicht gefunden. Bitte installiere Rust: https://rustup.rs/"
        echo "  Oder führe aus: ./install-rust.sh"
        exit 1
    fi
fi

# PostgreSQL (optional)
if command -v psql &> /dev/null; then
    success "PostgreSQL installiert"
else
    info "PostgreSQL nicht gefunden (optional für Website)"
fi

echo ""
echo "================================"
echo ""

# Frage welche Komponenten installiert werden sollen
echo "Welche Komponenten möchtest du installieren?"
echo "1) Alles (Website + Desktop-App + CLI)"
echo "2) Nur Website"
echo "3) Nur Desktop-App"
echo "4) Nur CLI"
read -p "Auswahl (1-4): " CHOICE

case $CHOICE in
    1)
        INSTALL_WEBSITE=true
        INSTALL_DESKTOP=true
        INSTALL_CLI=true
        ;;
    2)
        INSTALL_WEBSITE=true
        INSTALL_DESKTOP=false
        INSTALL_CLI=false
        ;;
    3)
        INSTALL_WEBSITE=false
        INSTALL_DESKTOP=true
        INSTALL_CLI=false
        ;;
    4)
        INSTALL_WEBSITE=false
        INSTALL_DESKTOP=false
        INSTALL_CLI=true
        ;;
    *)
        error "Ungültige Auswahl"
        exit 1
        ;;
esac

echo ""

# Website Setup
if [ "$INSTALL_WEBSITE" = true ]; then
    echo "🌐 Website Setup..."
    echo ""
    
    cd website
    
    info "Installiere Dependencies..."
    npm install
    success "Dependencies installiert"
    
    # .env.local erstellen
    if [ ! -f .env.local ]; then
        info "Erstelle .env.local..."
        echo ""
        read -p "Möchtest du den interaktiven Setup verwenden? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cd ..
            chmod +x setup-env.sh
            ./setup-env.sh
            cd website
        else
            cp .env.local.example .env.local
            success ".env.local erstellt"
            echo ""
            echo "⚠️  WICHTIG: Bearbeite website/.env.local und füge deine Credentials ein!"
            echo "  Siehe: PAYMENT_SETUP.md für Anleitung"
            echo "  Oder führe aus: ./setup-env.sh"
            echo ""
        fi
    fi
    
    # Prisma Setup
    info "Setup Prisma..."
    npx prisma generate
    success "Prisma Client generiert"
    
    # Database Push
    info "Initialisiere Datenbank..."
    npx prisma db push --skip-generate
    success "Database Schema angewendet"
    
    info "Datenbank wurde erfolgreich initialisiert (✓ dev.db erstellt)"
    
    cd ..
    echo ""
fi

# Desktop-App Setup
if [ "$INSTALL_DESKTOP" = true ]; then
    echo "🖥️  Desktop-App Setup..."
    echo ""
    
    cd desktop-app
    
    info "Installiere Dependencies..."
    npm install
    success "Dependencies installiert"
    
    # ML Backend Setup
    info "Setup ML Backend..."
    cd ml_backend
    
    if [ -d "venv" ]; then
        info "Virtuelle Umgebung existiert bereits"
    else
        info "Erstelle virtuelle Umgebung..."
        python3 -m venv venv
        success "Virtuelle Umgebung erstellt"
    fi
    
    info "Installiere Python Dependencies..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    success "Python Dependencies installiert"
    deactivate
    
    cd ../..
    echo ""
fi

# CLI Setup
if [ "$INSTALL_CLI" = true ]; then
    echo "⌨️  CLI Setup..."
    echo ""
    
    cd cli
    
    info "Installiere CLI..."
    pip3 install -e .
    success "CLI installiert"
    
    info "Teste CLI..."
    if frametrain --help &> /dev/null; then
        success "CLI funktioniert"
    else
        error "CLI Test fehlgeschlagen"
    fi
    
    cd ..
    echo ""
fi

# Shared Module (wenn Website oder Desktop-App installiert)
if [ "$INSTALL_WEBSITE" = true ] || [ "$INSTALL_DESKTOP" = true ]; then
    echo "📦 Shared Module Setup..."
    echo ""
    
    cd shared
    
    info "Installiere Dependencies..."
    npm install
    success "Dependencies installiert"
    
    info "Build Shared Module..."
    npm run build
    success "Shared Module gebaut"
    
    cd ..
    echo ""
fi

# Mache Scripts ausführbar
info "Mache Scripts ausführbar..."
chmod +x setup.sh install-rust.sh start.sh stop.sh restart.sh status.sh
success "Scripts sind ausführbar"

echo ""
# Zusammenfassung
echo "================================"
echo ""
echo "✨ Setup abgeschlossen!"
echo ""

if [ "$INSTALL_WEBSITE" = true ]; then
    echo "📱 Website starten:"
    echo "   cd website && npm run dev"
    echo "   → http://localhost:5001"
    echo ""
fi

if [ "$INSTALL_DESKTOP" = true ]; then
    echo "🖥️  Desktop-App starten:"
    echo "   cd desktop-app && npm run tauri:dev"
    echo ""
fi

if [ "$INSTALL_CLI" = true ]; then
    echo "⌨️  CLI verwenden:"
    echo "   frametrain --help"
    echo ""
fi

echo "📚 Weitere Infos:"
echo "   → Development Guide: docs/DEVELOPMENT.md"
echo "   → API Docs: docs/API.md"
echo "   → Deployment: docs/DEPLOYMENT.md"
echo ""

echo "🎉 Viel Erfolg mit FrameTrain!"
