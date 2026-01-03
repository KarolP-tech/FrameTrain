#!/bin/bash

# CRITICAL FIX - Tauri Auto-Signing Implementation
set -e

cd /Users/karol/Desktop/Laufende_Projekte/FrameTrain

echo "🚨 CRITICAL FIX: Tauri Auto-Signing Implementation"
echo ""
echo "Dies ist der RICHTIGE Fix für das Update-System!"
echo ""

# Status
git status --short
echo ""

# Add files
git add -A

# Commit
git commit -m "fix: Implement Tauri automatic signing (THE RIGHT WAY)

PROBLEM IDENTIFIED:
- We were trying to manually create tar.gz and sign them
- This is WRONG - Tauri does this automatically!
- The error 'Invalid symbol 46, offset 52' was because:
  * We were passing the key wrong
  * We shouldn't pass it manually at all

THE RIGHT WAY (Tauri Official):
1. Set TAURI_SIGNING_PRIVATE_KEY environment variable
2. Set TAURI_SIGNING_PRIVATE_KEY_PASSWORD environment variable  
3. Set createUpdaterArtifacts: true in tauri.conf.json
4. Run 'tauri build' - Tauri handles EVERYTHING automatically:
   - Creates .tar.gz/.zip update bundles
   - Signs them with minisign
   - Creates .sig signature files
   
CHANGES MADE:
✅ Added createUpdaterArtifacts: true to tauri.conf.json
✅ Removed ALL manual tar/zip/signing steps
✅ Removed npx tauri signer sign commands
✅ Removed temporary key file creation
✅ Let Tauri handle everything automatically

FILES CHANGED:
- .github/workflows/release-desktop-app.yml (completely rewritten)
- desktop-app2/src-tauri/tauri.conf.json (added createUpdaterArtifacts)
- CRITICAL_KEY_REGENERATION.md (instructions for fixing keys)

NEXT STEPS REQUIRED:
⚠️ Your GitHub Secret key is INVALID/CORRUPTED
⚠️ Follow CRITICAL_KEY_REGENERATION.md to:
   1. Generate new key pair
   2. Update GitHub Secrets
   3. Update pubkey in tauri.conf.json
   4. Test locally
   5. Push & test in CI

HOW IT WORKS NOW:
1. Developer runs: tauri build (with ENV vars set)
2. Tauri automatically:
   - Creates update bundles (.tar.gz/.zip)
   - Signs them with private key
   - Creates .sig files
3. GitHub Actions:
   - Just builds (Tauri signs automatically)
   - Uploads all artifacts
   - Generates latest.json from .sig files
4. Users:
   - App checks latest.json
   - Downloads signed update
   - Verifies signature
   - Installs if valid

This is the official Tauri way. No hacks, no workarounds.

Status: FUNDAMENTAL FIX - Requires key regeneration" || echo "Nothing to commit"

# Push
git push origin main

echo ""
echo "✅ Code gepusht!"
echo ""
echo "🚨 WICHTIG: GitHub Secrets müssen noch aktualisiert werden!"
echo ""
echo "📋 NÄCHSTE SCHRITTE (IN DIESER REIHENFOLGE):"
echo ""
echo "1️⃣  LIES: CRITICAL_KEY_REGENERATION.md"
echo "2️⃣  Generiere neue Keys: npm run tauri -- signer generate"
echo "3️⃣  Update GitHub Secrets (TAURI_SIGNING_PRIVATE_KEY)"
echo "4️⃣  Update tauri.conf.json (pubkey)"
echo "5️⃣  Test lokal: npm run tauri:build"
echo "6️⃣  Commit pubkey Änderung"
echo "7️⃣  Push"
echo "8️⃣  Trigger GitHub Actions Release"
echo ""
echo "❗ Ohne neue Keys wird der Build weiterhin fehlschlagen!"
echo ""
echo "🔗 https://github.com/KarolP-tech/FrameTrain"
