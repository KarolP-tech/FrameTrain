#!/bin/bash

# FrameTrain Desktop App - Test Script

echo "🧪 Testing Desktop App Login..."
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "   Create it with:"
    echo "   echo 'SUPABASE_URL=your_connection_string' > .env"
    exit 1
fi

echo "✅ .env file found"
echo ""

# Check if SUPABASE_URL is set
source .env
if [ -z "$SUPABASE_URL" ]; then
    echo "❌ SUPABASE_URL not set in .env"
    exit 1
fi

echo "✅ SUPABASE_URL is set"
echo ""

echo "📋 Starting app in dev mode..."
echo "   Watch the terminal for debug output!"
echo ""

npm run tauri:dev
