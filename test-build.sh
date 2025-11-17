#!/bin/bash
# TypeScript Build Fix

echo "🔧 Behebe TypeScript Build-Fehler..."
echo ""

cd "$(dirname "$0")/desktop-app"

echo "Versuche Build..."
npm run build

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build erfolgreich!"
    echo ""
    echo "Nächster Schritt:"
    echo "  npm run tauri:build"
else
    echo ""
    echo "❌ Build fehlgeschlagen"
    echo ""
    echo "Mögliche Lösungen:"
    echo "1. Node modules neu installieren:"
    echo "   rm -rf node_modules package-lock.json"
    echo "   npm install"
    echo ""
    echo "2. TypeScript Config prüfen:"
    echo "   cat tsconfig.json"
fi
