#!/bin/bash

# Quick Reset Script (no confirmation)
# Use this for rapid development testing

set -e

# Determine OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    APP_DATA_DIR="$HOME/Library/Application Support/com.frametrain.desktop2"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    APP_DATA_DIR="$HOME/.local/share/com.frametrain.desktop2"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    APP_DATA_DIR="$APPDATA/com.frametrain.desktop2"
else
    echo "❌ Unsupported OS"
    exit 1
fi

echo "🧹 Quick reset..."

# Delete everything
rm -rf "$APP_DATA_DIR"

echo "✅ Done! App data cleared."
