#!/bin/bash

# Macht alle Scripts im FrameTrain-Projekt ausführbar

cd "$(dirname "$0")"

echo "🔧 Mache Scripts ausführbar..."
echo ""

chmod +x *.sh
chmod +x website/*.sh 2>/dev/null || true
chmod +x desktop-app/*.sh 2>/dev/null || true
chmod +x cli/*.sh 2>/dev/null || true

echo "✓ Scripts sind jetzt ausführbar:"
echo ""
ls -1 *.sh | while read script; do
    echo "  • $script"
done

echo ""
echo "✓ Fertig!"
