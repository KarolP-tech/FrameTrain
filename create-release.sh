#!/bin/bash
# Erster Release Helper für FrameTrain
# Führt alle Schritte für den ersten Release aus

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🚀 FrameTrain - Erster Release"
echo "=============================="
echo ""

# Version aus package.json lesen
VERSION=$(grep '"version"' desktop-app/package.json | head -1 | awk -F'"' '{print $4}')

echo "📦 Version: v$VERSION"
echo ""

# Schritt 1: Pre-Release Check
echo "📋 Schritt 1/4: Pre-Release Check..."
if ./pre-release-check.sh; then
    echo "✅ Pre-Release Check erfolgreich"
else
    echo "❌ Pre-Release Check fehlgeschlagen!"
    echo ""
    echo "Behebe die Fehler und versuche es erneut."
    exit 1
fi

echo ""
read -p "Fortfahren? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Abgebrochen."
    exit 1
fi

# Schritt 2: Icons prüfen
echo ""
echo "🎨 Schritt 2/4: Icons prüfen..."
ICON_DIR="desktop-app/src-tauri/icons"

if [ ! -f "$ICON_DIR/icon.icns" ] || [ ! -f "$ICON_DIR/icon.ico" ]; then
    echo "⚠️  Icons fehlen!"
    read -p "Placeholder-Icons generieren? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$ICON_DIR"
        chmod +x generate-placeholder.sh
        ./generate-placeholder.sh
        cd "$SCRIPT_DIR"
        echo "✅ Icons generiert"
    else
        echo "❌ Icons erforderlich für Release!"
        exit 1
    fi
else
    echo "✅ Icons vorhanden"
fi

# Schritt 3: Git committen
echo ""
echo "📝 Schritt 3/4: Änderungen committen..."

if git diff-index --quiet HEAD --; then
    echo "✅ Keine Änderungen zu committen"
else
    echo "Änderungen gefunden:"
    git status --short
    echo ""
    read -p "Änderungen committen? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Commit Message: " COMMIT_MSG
        git add .
        git commit -m "$COMMIT_MSG"
        echo "✅ Änderungen committed"
    else
        echo "⚠️  Änderungen nicht committed"
    fi
fi

# Schritt 4: Tag erstellen und pushen
echo ""
echo "🏷️  Schritt 4/4: Release Tag erstellen..."
echo ""
echo "Dies wird:"
echo "  1. Git Tag 'v$VERSION' erstellen"
echo "  2. Tag zu GitHub pushen"
echo "  3. GitHub Actions starten (Build: 15-30 Min)"
echo ""
read -p "Fortfahren? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Abgebrochen."
    exit 1
fi

# Tag erstellen
if git tag "v$VERSION"; then
    echo "✅ Tag 'v$VERSION' erstellt"
else
    echo "⚠️  Tag existiert bereits oder Fehler"
    read -p "Tag überschreiben? (VORSICHT!) [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git tag -d "v$VERSION"
        git tag "v$VERSION"
        echo "✅ Tag 'v$VERSION' neu erstellt"
    else
        echo "Abgebrochen."
        exit 1
    fi
fi

# Main branch pushen
echo ""
echo "📤 Pushe main branch..."
git push origin main
echo "✅ Main branch gepusht"

# Tag pushen
echo ""
echo "📤 Pushe Tag (startet GitHub Actions)..."
git push origin "v$VERSION"
echo "✅ Tag gepusht"

# Fertig!
echo ""
echo "================================"
echo "🎉 Release gestartet!"
echo "================================"
echo ""
echo "📊 GitHub Actions:"
echo "   https://github.com/KarolP-tech/FrameTrain/actions"
echo ""
echo "⏱️  Build dauert ca. 15-30 Minuten"
echo ""
echo "Nach erfolgreichem Build:"
echo "   https://github.com/KarolP-tech/FrameTrain/releases/tag/v$VERSION"
echo ""
echo "📥 Installer werden verfügbar sein:"
echo "   - Windows: frametrain-windows-x86_64.msi"
echo "   - macOS:   frametrain-macos-universal.dmg"
echo "   - Linux:   frametrain-linux-x86_64.AppImage"
echo ""
echo "🔔 Du kannst den Build-Status in GitHub Actions verfolgen!"
echo ""
