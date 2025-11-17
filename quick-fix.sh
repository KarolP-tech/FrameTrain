#!/bin/bash
# Quick Fix Script - Behebt häufige Build-Probleme

echo "🔧 FrameTrain Quick Fix"
echo "======================="
echo ""

cd "$(dirname "$0")/desktop-app"

echo "1️⃣  Prüfe tsconfig.node.json..."
if [ ! -f "tsconfig.node.json" ]; then
    echo "   ❌ Fehlt - wird erstellt..."
    cat > tsconfig.node.json << 'EOF'
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
EOF
    echo "   ✅ Erstellt"
else
    echo "   ✅ Vorhanden"
fi

echo ""
echo "2️⃣  Prüfe Cargo.toml Tauri Version..."
if grep -q 'version = "2.9"' src-tauri/Cargo.toml; then
    echo "   ❌ Ungültige Version 2.9 gefunden - wird korrigiert..."
    sed -i.bak 's/version = "2.9"/version = "2"/g' src-tauri/Cargo.toml
    rm -f src-tauri/Cargo.toml.bak
    echo "   ✅ Korrigiert zu Version 2"
else
    echo "   ✅ Version korrekt"
fi

echo ""
echo "3️⃣  Prüfe Bundle Identifier..."
if grep -q '"com.frametrain.app"' src-tauri/tauri.conf.json; then
    echo "   ⚠️  Bundle ID endet mit .app - wird korrigiert..."
    sed -i.bak 's/"com.frametrain.app"/"com.frametrain.desktop"/g' src-tauri/tauri.conf.json
    rm -f src-tauri/tauri.conf.json.bak
    echo "   ✅ Korrigiert zu com.frametrain.desktop"
else
    echo "   ✅ Bundle ID korrekt"
fi

echo ""
echo "4️⃣  Prüfe node_modules..."
if [ ! -d "node_modules" ]; then
    echo "   ❌ node_modules fehlt - wird installiert..."
    npm install
    echo "   ✅ Installiert"
else
    echo "   ✅ Vorhanden"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "✅ Quick Fix abgeschlossen!"
echo ""
echo "Nächster Schritt:"
echo "  npm run build && npm run tauri:build"
echo ""
