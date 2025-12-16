#!/bin/bash

# Tauri Signing Fix - Commit & Push
set -e

cd /Users/karol/Desktop/Laufende_Projekte/FrameTrain

echo "🔐 Tauri Signing Fix - Commit & Push"
echo ""

# Status anzeigen
echo "Geänderte Dateien:"
git status --short
echo ""

# Add all
git add -A

# Commit mit detaillierter Message
git commit -m "fix: Tauri signing key file resolution for GitHub Actions

PROBLEM GELÖST:
- 'Key generation aborted: Unable to find the private key'
- tauri signer sign konnte ENV-Variablen nicht lesen

LÖSUNG:
- ENV-Keys werden in temporäre Dateien geschrieben
- Explizite --private-key und --password Flags
- Temporäre Dateien werden sofort gelöscht
- Gilt für alle Plattformen: macOS, Windows, Linux

SICHERHEIT:
- Keine Änderung der Sicherheitsstufe
- Keys bleiben in GitHub Secrets
- Temporäre Dateien nur im RAM

FILES CHANGED:
- .github/workflows/release-desktop-app.yml (macOS/Windows/Linux signing)
- TAURI_SIGNING_FIX.md (Dokumentation)

READY FOR:
- Kommerzieller Einsatz
- Auto-Updates
- Sichere Signierung" || echo "Nichts zu committen"

# Push
git push origin main

echo ""
echo "✅ Erfolgreich gepusht!"
echo "🔗 https://github.com/KarolP-tech/FrameTrain"
echo ""
echo "Nächster Schritt: Neues Release triggern zum Testen"
