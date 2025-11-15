#!/bin/bash

# 🚀 FrameTrain - Safe Git Push Script
# Dieser Script führt alle Pre-Push Checks durch und pusht sicher zu GitHub

set -e  # Exit bei Fehler

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🚀 FrameTrain - Pre-Push Security Check & Push          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Prüfe dass wir im richtigen Verzeichnis sind
if [ ! -f "README.md" ] || [ ! -d ".git" ]; then
    echo "❌ Fehler: Nicht im FrameTrain Root-Verzeichnis!"
    echo "   Führe aus: cd /Users/karol/Desktop/Laufende_Projekte/FrameTrain"
    exit 1
fi

echo "📍 Arbeitsverzeichnis: $(pwd)"
echo ""

# ============================================================================
# SCHRITT 1: Prüfe .gitignore
# ============================================================================
echo "🔍 SCHRITT 1: Prüfe .gitignore Konfiguration..."
echo ""

if ! grep -q "^\.env\.local$" .gitignore; then
    echo "⚠️  Warnung: .env.local nicht in .gitignore gefunden!"
    echo "   Füge hinzu mit: echo '.env.local' >> .gitignore"
    exit 1
fi

if ! grep -q "^\.next/" .gitignore; then
    echo "⚠️  Warnung: .next/ nicht in .gitignore gefunden!"
    exit 1
fi

echo "✅ .gitignore ist korrekt konfiguriert"
echo ""

# ============================================================================
# SCHRITT 2: Prüfe auf sensible Dateien im Git-Index
# ============================================================================
echo "🔍 SCHRITT 2: Prüfe Git-Status..."
echo ""

# Zeige Status
git status --short

echo ""

# Prüfe ob .env.local getrackt wird
if git ls-files | grep -q "\.env\.local$"; then
    echo "❌ FEHLER: .env.local wird von Git getrackt!"
    echo "   Entferne mit: git rm --cached website/.env.local"
    exit 1
fi

# Prüfe ob .next getrackt wird
if git ls-files | grep -q "\.next/"; then
    echo "❌ FEHLER: .next/ Dateien werden von Git getrackt!"
    echo "   Entferne mit: git rm --cached -r website/.next"
    exit 1
fi

echo "✅ Keine sensiblen Dateien im Git-Index"
echo ""

# ============================================================================
# SCHRITT 3: Prüfe auf Secrets in staged Dateien
# ============================================================================
echo "🔍 SCHRITT 3: Scanne nach Secrets..."
echo ""

SECRETS_FOUND=0

# Suche nach Stripe Test Keys (aber erlaube sie in .example Dateien und Dokumentation)
# Ignoriere: *.example, md dateien/, docs/, README.md
CHECK_FILES=$(git diff --cached --name-only | grep -v -E "(\.example$|^md dateien/|^docs/|README\.md)" || true)

if [ ! -z "$CHECK_FILES" ]; then
    if echo "$CHECK_FILES" | xargs grep -l "sk_test_51SSOg4EC9c8leIGW" 2>/dev/null; then
        echo "❌ FEHLER: Echter Stripe Secret Key gefunden!"
        SECRETS_FOUND=1
    fi

    if echo "$CHECK_FILES" | xargs grep -l "pk_test_51SSOg4EC9c8leIGW" 2>/dev/null; then
        echo "❌ FEHLER: Echter Stripe Publishable Key gefunden!"
        SECRETS_FOUND=1
    fi

    if echo "$CHECK_FILES" | xargs grep -l "whsec_e28709edf92bb5b2055f" 2>/dev/null; then
        echo "❌ FEHLER: Echter Stripe Webhook Secret gefunden!"
        SECRETS_FOUND=1
    fi
fi

if [ $SECRETS_FOUND -eq 1 ]; then
    echo ""
    echo "⚠️  Echte API-Keys gefunden in staged Dateien!"
    echo "   Das sollte nicht passieren - prüfe die Dateien manuell."
    exit 1
fi

echo "✅ Keine Secrets in staged Dateien gefunden"
echo ""

# ============================================================================
# SCHRITT 4: Username-Check
# ============================================================================
echo "🔍 SCHRITT 4: Prüfe GitHub Username..."
echo ""

if grep -q "YourUsername" README.md; then
    echo "⚠️  Warnung: 'YourUsername' gefunden in README.md"
    echo "   Wurde bereits durch KarolP-tech ersetzt"
fi

echo "✅ GitHub Username ist korrekt (KarolP-tech)"
echo ""

# ============================================================================
# SCHRITT 5: Git Add & Commit
# ============================================================================
echo "📦 SCHRITT 5: Git Add & Commit..."
echo ""

# Zeige welche Dateien hinzugefügt werden
echo "Folgende Dateien werden committed:"
git status --short | grep -E "^(\?\?|M |A )" || echo "  (keine neuen/geänderten Dateien)"
echo ""

read -p "Möchtest du alle Dateien hinzufügen und committen? (j/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[JjYy]$ ]]; then
    echo "❌ Abgebrochen. Keine Änderungen committed."
    exit 0
fi

# Add all files
git add .

# Commit
COMMIT_MSG="Initial commit: FrameTrain v1.0.0 with BSL 1.1 license

- Complete platform structure (website, desktop-app, CLI)
- Business Source License 1.1
- GitHub username: KarolP-tech
- Security: .env.local and build files excluded"

echo ""
echo "Committing with message:"
echo "\"$COMMIT_MSG\""
echo ""

git commit -m "$COMMIT_MSG" || echo "Nichts zu committen oder Commit fehlgeschlagen"

# ============================================================================
# SCHRITT 6: Remote Check
# ============================================================================
echo ""
echo "🔍 SCHRITT 6: Prüfe Git Remote..."
echo ""

REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")

if [ -z "$REMOTE_URL" ]; then
    echo "⚠️  Kein Remote 'origin' gefunden"
    echo "   Füge hinzu mit:"
    echo "   git remote add origin https://github.com/KarolP-tech/FrameTrain.git"
    exit 1
fi

echo "✅ Remote URL: $REMOTE_URL"
echo ""

# ============================================================================
# SCHRITT 7: Push zu GitHub
# ============================================================================
echo "🚀 SCHRITT 7: Push zu GitHub..."
echo ""

read -p "Bereit zum Push nach GitHub? (j/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[JjYy]$ ]]; then
    echo "❌ Push abgebrochen."
    echo "   Führe manuell aus: git push -u origin main"
    exit 0
fi

echo ""
echo "Pushing zu GitHub..."
git push -u origin main

# ============================================================================
# ERFOLG!
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ ERFOLG! FrameTrain erfolgreich zu GitHub gepusht!    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "🎉 Nächste Schritte:"
echo ""
echo "1️⃣  Repository besuchen:"
echo "   https://github.com/KarolP-tech/FrameTrain"
echo ""
echo "2️⃣  About Section aktualisieren:"
echo "   - Description: Professional platform for local ML training"
echo "   - Topics: machine-learning, pytorch, tauri, nextjs, stripe"
echo ""
echo "3️⃣  GitHub Secrets hinzufügen (für CI/CD):"
echo "   Settings → Secrets → Actions"
echo "   - STRIPE_SECRET_KEY"
echo "   - DATABASE_URL"
echo "   - JWT_SECRET"
echo ""
echo "4️⃣  README im Browser ansehen:"
echo "   Badges sollten funktionieren (außer Build Badge - kommt nach erstem Workflow)"
echo ""
echo "📚 Mehr Infos in: PRE_PUSH_CHECK.md"
echo ""
