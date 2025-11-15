#!/bin/bash

# Test Auth Flow Script

echo "🧪 Teste Auth-Flow..."
echo ""

# 1. Stoppe Services
echo "→ Stoppe Services..."
./stop.sh > /dev/null 2>&1

# 2. Starte Services
echo "→ Starte Services neu..."
./start.sh > /dev/null 2>&1
sleep 3

echo ""
echo "✅ Services laufen!"
echo ""
echo "📱 Teste jetzt die App:"
echo "  1. Öffne http://localhost:5001"
echo "  2. Registriere dich (falls noch nicht geschehen)"
echo "  3. Du solltest automatisch zum Dashboard weitergeleitet werden"
echo "  4. Der Header zeigt deine E-Mail und einen 'Abmelden' Button"
echo ""
echo "🔍 Logs verfolgen:"
echo "  tail -f .pids/website.log"
echo ""
