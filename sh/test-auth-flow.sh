#!/bin/bash

# Auth-Flow Test Script

echo "🧪 FrameTrain Auth-Flow Test"
echo "============================="
echo ""

echo "Dieser Test prüft:"
echo "  ✓ Automatische Weiterleitung nach Registrierung"
echo "  ✓ Dashboard-Zugriff für eingeloggte User"
echo "  ✓ Startseite leitet zum Dashboard um wenn eingeloggt"
echo "  ✓ Header zeigt korrekten Auth-Status"
echo ""

echo "📋 Test-Schritte:"
echo ""
echo "1️⃣  Öffne http://localhost:5001"
echo "    → Sollte Landing Page zeigen (wenn nicht eingeloggt)"
echo "    → Sollte zum Dashboard weiterleiten (wenn eingeloggt)"
echo ""
echo "2️⃣  Registriere dich mit neuer E-Mail"
echo "    → Nach Registrierung: Auto-Login"
echo "    → Automatische Weiterleitung zum Dashboard"
echo "    → Header zeigt E-Mail + Abmelden-Button"
echo ""
echo "3️⃣  Gehe zu http://localhost:5001"
echo "    → Sollte direkt zum Dashboard weiterleiten"
echo "    → NICHT Landing Page zeigen"
echo ""
echo "4️⃣  Klicke 'Abmelden' im Header"
echo "    → Weiterleitung zur Startseite"
echo "    → Header zeigt Login + Starten"
echo ""
echo "5️⃣  Klicke 'Dashboard' ohne Login"
echo "    → Sollte zur Login-Seite weiterleiten"
echo ""

echo "🔍 Debugging:"
echo ""
echo "Browser-Console öffnen (F12) und prüfen:"
echo "  → Auth-Status: fetch('/api/auth/me', {credentials: 'include'}).then(r => r.json()).then(console.log)"
echo "  → Cookie: document.cookie"
echo ""

echo "📊 Logs verfolgen:"
echo "  tail -f .pids/website.log"
echo ""

read -p "Test starten? Browser wird geöffnet (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "→ Öffne Browser..."
    open http://localhost:5001 2>/dev/null || xdg-open http://localhost:5001 2>/dev/null || echo "Öffne manuell: http://localhost:5001"
    echo ""
    echo "✅ Browser geöffnet!"
    echo ""
    echo "Folge den Test-Schritten oben ☝️"
fi
