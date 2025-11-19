#!/bin/bash

# Desktop App - Create .env with properly encoded password

echo "🔧 Creating .env file with proper encoding..."

# Your Supabase password has special characters: gD7eT4iP9$^
# These need to be URL-encoded:
# $ = %24
# ^ = %5E

cat > .env << 'EOF'
SUPABASE_URL=postgresql://postgres.pmilxbuzfghbphjjaiar:gD7eT4iP9%24%5E@aws-1-eu-west-1.pooler.supabase.com:6543/postgres
EOF

echo "✅ .env file created!"
echo ""
echo "Content:"
cat .env
echo ""
echo "🚀 Now run: npm run tauri:dev"
