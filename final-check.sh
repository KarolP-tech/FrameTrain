#!/bin/bash
# Finale Checkliste vor dem ersten Release

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║  ✅ FrameTrain - Pre-Release Checkliste                   ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Farben
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_item() {
    local description=$1
    local command=$2
    
    echo -n "[ ] $description..."
    if eval "$command" &> /dev/null; then
        echo -e " ${GREEN}✅${NC}"
        return 0
    else
        echo -e " ${RED}❌${NC}"
        return 1
    fi
}

warn_item() {
    local description=$1
    echo -e "${YELLOW}⚠️  $description${NC}"
}

info_item() {
    local description=$1
    echo -e "ℹ️  $description"
}

ERRORS=0

echo "1️⃣  REPOSITORY"
echo "─────────────────────────────────────────────────────────────"
check_item "Git Repository vorhanden" "[ -d .git ]" || ((ERRORS++))
check_item "Git Remote konfiguriert" "git remote get-url origin" || ((ERRORS++))
check_item "Keine uncommitted Changes" "git diff-index --quiet HEAD --" || warn_item "Uncommitted Changes vorhanden"
echo ""

echo "2️⃣  VERSIONEN"
echo "─────────────────────────────────────────────────────────────"
if [ -f "desktop-app/package.json" ]; then
    PKG_VERSION=$(grep '"version"' desktop-app/package.json | head -1 | awk -F'"' '{print $4}')
    info_item "desktop-app/package.json: v$PKG_VERSION"
fi
if [ -f "desktop-app/src-tauri/Cargo.toml" ]; then
    CARGO_VERSION=$(grep '^version' desktop-app/src-tauri/Cargo.toml | head -1 | awk -F'"' '{print $2}')
    info_item "Cargo.toml: v$CARGO_VERSION"
fi
if [ -f "desktop-app/src-tauri/tauri.conf.json" ]; then
    TAURI_VERSION=$(grep '"version"' desktop-app/src-tauri/tauri.conf.json | head -1 | awk -F'"' '{print $4}')
    info_item "tauri.conf.json: v$TAURI_VERSION"
fi

if [ "$PKG_VERSION" = "$CARGO_VERSION" ] && [ "$PKG_VERSION" = "$TAURI_VERSION" ]; then
    echo -e "${GREEN}✅ Alle Versionen identisch: v$PKG_VERSION${NC}"
else
    echo -e "${RED}❌ Versionen nicht identisch!${NC}"
    ((ERRORS++))
fi
echo ""

echo "3️⃣  ICONS"
echo "─────────────────────────────────────────────────────────────"
ICON_DIR="desktop-app/src-tauri/icons"
check_item "32x32.png" "[ -f $ICON_DIR/32x32.png ]" || ((ERRORS++))
check_item "128x128.png" "[ -f $ICON_DIR/128x128.png ]" || ((ERRORS++))
check_item "128x128@2x.png" "[ -f $ICON_DIR/128x128@2x.png ]" || ((ERRORS++))
check_item "icon.icns (macOS)" "[ -f $ICON_DIR/icon.icns ]" || ((ERRORS++))
check_item "icon.ico (Windows)" "[ -f $ICON_DIR/icon.ico ]" || ((ERRORS++))
echo ""

echo "4️⃣  DEPENDENCIES"
echo "─────────────────────────────────────────────────────────────"
check_item "Node.js" "command -v node" || ((ERRORS++))
check_item "npm" "command -v npm" || ((ERRORS++))
check_item "Rust" "command -v rustc" || ((ERRORS++))
check_item "Cargo" "command -v cargo" || ((ERRORS++))
check_item "Python3" "command -v python3" || ((ERRORS++))
echo ""

echo "5️⃣  GITHUB ACTIONS"
echo "─────────────────────────────────────────────────────────────"
check_item "release.yml" "[ -f .github/workflows/release.yml ]" || ((ERRORS++))
check_item "build-desktop.yml" "[ -f .github/workflows/build-desktop.yml ]" || warn_item "Optional"
echo ""

echo "6️⃣  WEBSITE"
echo "─────────────────────────────────────────────────────────────"
info_item "Website gehostet auf Vercel? (Manuell prüfen)"
info_item "Supabase DB verbunden? (Manuell prüfen)"
info_item "Stripe eingerichtet? (Manuell prüfen)"
echo ""

echo "═════════════════════════════════════════════════════════════"

if [ $ERRORS -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                            ║${NC}"
    echo -e "${GREEN}║  🎉  ALLES BEREIT FÜR DEN RELEASE!                        ║${NC}"
    echo -e "${GREEN}║                                                            ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Nächster Schritt:"
    echo ""
    echo "  ./create-release.sh"
    echo ""
else
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                                                            ║${NC}"
    echo -e "${RED}║  ❌  $ERRORS FEHLER GEFUNDEN                                ║${NC}"
    echo -e "${RED}║                                                            ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Bitte behebe die Fehler:"
    echo ""
    
    if ! [ -f "$ICON_DIR/icon.icns" ] || ! [ -f "$ICON_DIR/icon.ico" ]; then
        echo "  Icons generieren:"
        echo "    cd desktop-app/src-tauri/icons"
        echo "    python3 generate-icons.py"
        echo ""
    fi
    
    if ! command -v rustc &> /dev/null; then
        echo "  Rust installieren:"
        echo "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        echo ""
    fi
fi
