#!/bin/bash

# Check welche Ports belegt sind
echo "Prüfe Ports 5000-5010..."
echo ""

for port in {5000..5010}; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        pid=$(lsof -Pi :$port -sTCP:LISTEN -t)
        process=$(ps -p $pid -o comm= 2>/dev/null)
        echo "❌ Port $port: BELEGT (PID: $pid, Prozess: $process)"
    else
        echo "✅ Port $port: FREI"
    fi
done

echo ""
echo "Empfehlung: Verwende den ersten freien Port für deine Services"
