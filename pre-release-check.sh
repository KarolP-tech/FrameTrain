#!/bin/bash
# Pre-Release Checklist für FrameTrain
# Prüft ob alles bereit ist für einen Release

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🔍 FrameTrain Pre-Release Check"
echo "================================"
echo ""

ERRORS=0
WARNINGS=0

# Funktion für Checks
check_ok() {
    echo "✅ $1"
}

check_error() {
    echo "❌ $1"
    ((ERRORS++))
}

check_warning() {
    echo "⚠️  $1"
    ((WARNINGS++))
}

# 1. Prüfe ob in FrameTrain Root
echo "📁 Repository Struktur..."
if [ -f "package.json" ] && [ -d "desktop-app" ] && [ -d "website" ]; then
    check_ok "Repository Struktur korrekt"
else
    check_error "Nicht im FrameTrain Root-Verzeichnis!"
    exit 1
fi

# 2. Prüfe Git Status
echo ""
echo "📊 Git Status..."
if [ -d ".git" ]; then
    check_ok "Git Repository vorhanden"
    
    # Uncommitted changes?
    if git diff-index --quiet HEAD --; then
        check_ok "Keine uncommitted Changes"
    else
        check_warning "Uncommitted Changes vorhanden - bitte committen"
    fi
    
    # Remote configured?
    if git remote get-url origin &> /dev/null; then
        REMOTE=$(git remote get-url origin)
        check_ok "Remote konfiguriert: $REMOTE"
    else
        check_error "Kein Git Remote konfiguriert!"
    fi
else
    check_error "Kein Git Repository!"
fi

# 3. Prüfe Versionen
echo ""
echo "🔢 Version Checks..."

# Desktop App package.json
if [ -f "desktop-app/package.json" ]; then
    PKG_VERSION=$(grep '"version"' desktop-app/package.json | head -1 | awk -F'"' '{print $4}')
    check_ok "desktop-app/package.json: v$PKG_VERSION"
else
    check_error "desktop-app/package.json nicht gefunden!"
fi

# Cargo.toml
if [ -f "desktop-app/src-tauri/Cargo.toml" ]; then
    CARGO_VERSION=$(grep '^version' desktop-app/src-tauri/Cargo.toml | head -1 | awk -F'"' '{print $2}')
    check_ok "Cargo.toml: v$CARGO_VERSION"
else
    check_error "desktop-app/src-tauri/Cargo.toml nicht gefunden!"
fi

# tauri.conf.json
if [ -f "desktop-app/src-tauri/tauri.conf.json" ]; then
    TAURI_VERSION=$(grep '"version"' desktop-app/src-tauri/tauri.conf.json | head -1 | awk -F'"' '{print $4}')
    check_ok "tauri.conf.json: v$TAURI_VERSION"
else
    check_error "desktop-app/src-tauri/tauri.conf.json nicht gefunden!"
fi

# Versionen identisch?
if [ "$PKG_VERSION" = "$CARGO_VERSION" ] && [ "$PKG_VERSION" = "$TAURI_VERSION" ]; then
    check_ok "Alle Versionen identisch: v$PKG_VERSION"
else
    check_error "Versionen nicht identisch! PKG:$PKG_VERSION CARGO:$CARGO_VERSION TAURI:$TAURI_VERSION"
fi

# 4. Prüfe Icons
echo ""
echo "🎨 Icon Checks..."
ICON_DIR="desktop-app/src-tauri/icons"

if [ -f "$ICON_DIR/32x32.png" ]; then
    check_ok "32x32.png vorhanden"
else
    check_error "32x32.png fehlt!"
fi

if [ -f "$ICON_DIR/128x128.png" ]; then
    check_ok "128x128.png vorhanden"
else
    check_error "128x128.png fehlt!"
fi

if [ -f "$ICON_DIR/128x128@2x.png" ]; then
    check_ok "128x128@2x.png vorhanden"
else
    check_error "128x128@2x.png fehlt!"
fi

if [ -f "$ICON_DIR/icon.icns" ]; then
    check_ok "icon.icns (macOS) vorhanden"
else
    check_error "icon.icns fehlt!"
fi

if [ -f "$ICON_DIR/icon.ico" ]; then
    check_ok "icon.ico (Windows) vorhanden"
else
    check_error "icon.ico fehlt!"
fi

# 5. Prüfe Dependencies
echo ""
echo "📦 Dependencies..."

# Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    check_ok "Node.js: $NODE_VERSION"
else
    check_error "Node.js nicht installiert!"
fi

# npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    check_ok "npm: v$NPM_VERSION"
else
    check_error "npm nicht installiert!"
fi

# Rust
if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version)
    check_ok "Rust: $RUST_VERSION"
else
    check_error "Rust nicht installiert!"
fi

# Cargo
if command -v cargo &> /dev/null; then
    CARGO_VERSION_CMD=$(cargo --version)
    check_ok "Cargo: $CARGO_VERSION_CMD"
else
    check_error "Cargo nicht installiert!"
fi

# Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    check_ok "Python: $PYTHON_VERSION"
else
    check_error "Python3 nicht installiert!"
fi

# 6. Prüfe GitHub Actions
echo ""
echo "⚙️  GitHub Actions..."

if [ -f ".github/workflows/release.yml" ]; then
    check_ok "release.yml vorhanden"
else
    check_error "release.yml fehlt!"
fi

if [ -f ".github/workflows/build-desktop.yml" ]; then
    check_ok "build-desktop.yml vorhanden"
else
    check_warning "build-desktop.yml fehlt (optional)"
fi

# 7. Prüfe .gitignore
echo ""
echo "🔒 Security Checks..."

if grep -q "\.env\.local" .gitignore; then
    check_ok ".env.local in .gitignore"
else
    check_error ".env.local nicht in .gitignore!"
fi

if grep -q "secrets/" .gitignore; then
    check_ok "secrets/ in .gitignore"
else
    check_warning "secrets/ nicht in .gitignore"
fi

# 8. Test ob Build funktioniert (optional)
echo ""
echo "🧪 Build Test (optional)..."
read -p "Lokalen Build testen? (dauert 5-10 Min) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔨 Starte Build..."
    cd desktop-app
    if npm run build && npm run tauri:build; then
        check_ok "Build erfolgreich!"
    else
        check_error "Build fehlgeschlagen!"
    fi
    cd ..
else
    check_warning "Build-Test übersprungen"
fi

# Zusammenfassung
echo ""
echo "================================"
echo "📊 Zusammenfassung"
echo "================================"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "🎉 Perfekt! Alles bereit für den Release!"
    echo ""
    echo "Nächster Schritt:"
    echo "  git tag v$PKG_VERSION"
    echo "  git push origin v$PKG_VERSION"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo "⚠️  $WARNINGS Warnungen, aber Release möglich"
    echo ""
    echo "Nächster Schritt:"
    echo "  git tag v$PKG_VERSION"
    echo "  git push origin v$PKG_VERSION"
    exit 0
else
    echo "❌ $ERRORS Fehler gefunden!"
    echo "⚠️  $WARNINGS Warnungen"
    echo ""
    echo "Bitte behebe die Fehler vor dem Release."
    exit 1
fi
