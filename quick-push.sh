#!/bin/bash

# Quick Push - Ohne Sicherheits-Checks (nur wenn du sicher bist!)
# Für normale Verwendung nutze: ./push.sh

set -e

echo "🚀 FrameTrain - Quick Push"
echo ""

cd /Users/karol/Desktop/Laufende_Projekte/FrameTrain

# Status
echo "Status:"
git status --short
echo ""

# Add all
git add .

# Commit
git commit -m "Initial commit: FrameTrain v1.0.0 with BSL 1.1 license" || echo "Nothing to commit"

# Push
git push -u origin main

echo ""
echo "✅ Fertig! https://github.com/KarolP-tech/FrameTrain"
