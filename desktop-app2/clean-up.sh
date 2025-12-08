#!/bin/bash

# FrameTrain Complete Cleanup Script
# Löscht ALLE gespeicherten Daten (Modelle, Datasets, Datenbank, Logs, Training-Outputs)

set -e  # Exit on error

echo "================================================"
echo "  FrameTrain Complete Cleanup Script"
echo "================================================"
echo ""
echo "⚠️  WARNUNG: Dieses Script löscht ALLE Daten!"
echo ""
echo "Folgende Daten werden gelöscht:"
echo "  - Alle Modelle und Versionen"
echo "  - Alle Datasets"
echo "  - Die gesamte Datenbank"
echo "  - Alle Training-Outputs und Logs"
echo "  - Alle Checkpoints"
echo "  - Metadata und Konfigurationen"
echo ""
read -p "Möchtest du wirklich fortfahren? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Abgebrochen."
    exit 0
fi

echo ""
echo "🧹 Starte Cleanup..."
echo ""

# Detect OS and set app data directory
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    APP_DATA_DIR="$HOME/Library/Application Support/com.frametrain.desktop2"
    echo "📍 macOS erkannt"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    APP_DATA_DIR="$HOME/.local/share/com.frametrain.desktop2"
    echo "📍 Linux erkannt"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash)
    APP_DATA_DIR="$APPDATA/com.frametrain.desktop2"
    echo "📍 Windows erkannt"
else
    echo "❌ Unbekanntes Betriebssystem: $OSTYPE"
    exit 1
fi

echo "📂 App Data Directory: $APP_DATA_DIR"
echo ""

# Check if directory exists
if [ ! -d "$APP_DATA_DIR" ]; then
    echo "⚠️  App Data Directory existiert nicht."
    echo "    Nichts zu löschen!"
    exit 0
fi

# Function to delete directory with feedback
delete_dir() {
    local dir=$1
    local name=$2
    
    if [ -d "$dir" ]; then
        echo "🗑️  Lösche $name..."
        rm -rf "$dir"
        echo "   ✅ $name gelöscht"
    else
        echo "   ⏭️  $name existiert nicht"
    fi
}

# Function to delete file with feedback
delete_file() {
    local file=$1
    local name=$2
    
    if [ -f "$file" ]; then
        echo "🗑️  Lösche $name..."
        rm -f "$file"
        echo "   ✅ $name gelöscht"
    else
        echo "   ⏭️  $name existiert nicht"
    fi
}

# 1. Delete Models Directory
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Modelle"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
delete_dir "$APP_DATA_DIR/models" "Modelle-Ordner"
echo ""

# 2. Delete Training Outputs
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Training-Outputs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
delete_dir "$APP_DATA_DIR/training_outputs" "Training-Outputs"
echo ""

# 3. Delete Checkpoints
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. Checkpoints"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
delete_dir "$APP_DATA_DIR/checkpoints" "Checkpoints"
echo ""

# 4. Delete Datasets (if stored separately)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. Datasets"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
delete_dir "$APP_DATA_DIR/datasets" "Datasets-Ordner"
echo ""

# 5. Delete Database
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. Datenbank"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
delete_file "$APP_DATA_DIR/frametrain.db" "Datenbank (frametrain.db)"
delete_file "$APP_DATA_DIR/frametrain.db-shm" "Datenbank-Shared Memory"
delete_file "$APP_DATA_DIR/frametrain.db-wal" "Datenbank-Write Ahead Log"
echo ""

# 6. Delete Logs
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. Logs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
delete_dir "$APP_DATA_DIR/logs" "Logs-Ordner"
delete_file "$APP_DATA_DIR/training_jobs.json" "Training Jobs JSON"
echo ""

# 7. Delete Metadata and Config
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7. Metadata & Konfiguration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
delete_file "$APP_DATA_DIR/models_metadata.json" "Models Metadata"
delete_file "$APP_DATA_DIR/app_config.json" "App Konfiguration"
delete_dir "$APP_DATA_DIR/cache" "Cache-Ordner"
delete_dir "$APP_DATA_DIR/temp" "Temp-Ordner"
echo ""

# 8. Calculate space freed
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "8. Statistiken"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d "$APP_DATA_DIR" ]; then
    remaining_size=$(du -sh "$APP_DATA_DIR" 2>/dev/null | cut -f1 || echo "0B")
    remaining_files=$(find "$APP_DATA_DIR" -type f 2>/dev/null | wc -l || echo "0")
    echo "📊 Verbleibende Daten: $remaining_size ($remaining_files Dateien)"
    
    if [ "$remaining_files" -eq 0 ]; then
        echo ""
        echo "🗑️  Lösche leeren App Data Ordner..."
        rmdir "$APP_DATA_DIR" 2>/dev/null || true
        echo "   ✅ App Data Ordner gelöscht"
    fi
else
    echo "✅ Alle Daten gelöscht!"
fi

echo ""
echo "================================================"
echo "  ✅ Cleanup abgeschlossen!"
echo "================================================"
echo ""
echo "Alle FrameTrain-Daten wurden erfolgreich entfernt."
echo "Du kannst die App jetzt neu starten für einen"
echo "frischen Start."
echo ""