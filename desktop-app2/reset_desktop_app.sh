#!/bin/bash

# FrameTrain Desktop App - Complete Reset Script
# This script resets the app to factory defaults by deleting all user data

set -e  # Exit on error

echo "================================================"
echo "🔄 FrameTrain Desktop App - COMPLETE RESET"
echo "================================================"
echo ""
echo "⚠️  WARNING: This will DELETE ALL DATA:"
echo "  - All models"
echo "  - All datasets"
echo "  - All training results"
echo "  - All test results"
echo "  - All configurations"
echo "  - The entire database"
echo ""
echo "This action CANNOT be undone!"
echo ""
read -p "Are you sure you want to continue? (type 'YES' to confirm): " confirmation

if [ "$confirmation" != "YES" ]; then
    echo "❌ Reset cancelled."
    exit 0
fi

echo ""
echo "🚀 Starting reset process..."
echo ""

# Determine the app data directory based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    APP_DATA_DIR="$HOME/Library/Application Support/com.frametrain.desktop2"
    APP_CONFIG_DIR="$HOME/Library/Application Support/com.frametrain.desktop2"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    APP_DATA_DIR="$HOME/.local/share/com.frametrain.desktop2"
    APP_CONFIG_DIR="$HOME/.config/com.frametrain.desktop2"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash/MSYS)
    APP_DATA_DIR="$APPDATA/com.frametrain.desktop2"
    APP_CONFIG_DIR="$APPDATA/com.frametrain.desktop2"
else
    echo "❌ Unsupported operating system: $OSTYPE"
    exit 1
fi

echo "📁 App Data Directory: $APP_DATA_DIR"
echo "📁 App Config Directory: $APP_CONFIG_DIR"
echo ""

# Function to safely delete directory
safe_delete() {
    local dir=$1
    local name=$2
    
    if [ -d "$dir" ]; then
        echo "🗑️  Deleting $name..."
        rm -rf "$dir"
        echo "✅ $name deleted"
    else
        echo "ℹ️  $name not found (already clean)"
    fi
}

# Function to safely delete file
safe_delete_file() {
    local file=$1
    local name=$2
    
    if [ -f "$file" ]; then
        echo "🗑️  Deleting $name..."
        rm -f "$file"
        echo "✅ $name deleted"
    else
        echo "ℹ️  $name not found (already clean)"
    fi
}

echo "🧹 Cleaning application data..."
echo ""

# 1. Delete Database
safe_delete_file "$APP_DATA_DIR/frametrain.db" "Database"
safe_delete_file "$APP_DATA_DIR/frametrain.db-shm" "Database (shared memory)"
safe_delete_file "$APP_DATA_DIR/frametrain.db-wal" "Database (write-ahead log)"

# 2. Delete Models
safe_delete "$APP_DATA_DIR/models" "Models directory"

# 3. Delete Training Outputs
safe_delete "$APP_DATA_DIR/training_outputs" "Training outputs"

# 4. Delete Test Outputs
safe_delete "$APP_DATA_DIR/test_outputs" "Test outputs"

# 5. Delete Exports
safe_delete "$APP_DATA_DIR/exports" "Exports"

# 6. Delete Metadata Files
safe_delete_file "$APP_DATA_DIR/models_metadata.json" "Models metadata"
safe_delete_file "$APP_DATA_DIR/datasets_metadata.json" "Datasets metadata"
safe_delete_file "$APP_DATA_DIR/test_jobs.json" "Test jobs history"

# 7. Delete Config Files
safe_delete_file "$APP_CONFIG_DIR/config.json" "User configuration"

# 8. Delete Logs (if any)
safe_delete "$APP_DATA_DIR/logs" "Logs directory"

# 9. Delete temporary files
safe_delete "$APP_DATA_DIR/tmp" "Temporary files"
safe_delete "$APP_DATA_DIR/cache" "Cache"

echo ""
echo "================================================"
echo "✅ RESET COMPLETE!"
echo "================================================"
echo ""
echo "The app has been reset to factory defaults."
echo "All user data has been deleted."
echo ""
echo "You can now start the app fresh by running:"
echo "  cd desktop-app2"
echo "  npm run tauri dev"
echo ""
echo "You will need to:"
echo "  1. Accept the plugin installation prompt (first launch)"
echo "  2. Log in with your API key"
echo "  3. Import/download models again"
echo ""
