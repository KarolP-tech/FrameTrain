#!/bin/bash

# Test verschiedene Connection String Formate

echo "🧪 Testing different connection string formats..."
echo ""

# Format 1: Basic
echo "Format 1: Basic with encoded password"
CONN1="postgresql://postgres.pmilxbuzfghbphjjaiar:gD7eT4iP9%24%5E@aws-1-eu-west-1.pooler.supabase.com:6543/postgres"
echo "$CONN1"
echo ""

# Format 2: With pgbouncer
echo "Format 2: With pgbouncer parameter"
CONN2="postgresql://postgres.pmilxbuzfghbphjjaiar:gD7eT4iP9%24%5E@aws-1-eu-west-1.pooler.supabase.com:6543/postgres?pgbouncer=true"
echo "$CONN2"
echo ""

# Format 3: With connection_limit
echo "Format 3: With connection_limit"
CONN3="postgresql://postgres.pmilxbuzfghbphjjaiar:gD7eT4iP9%24%5E@aws-1-eu-west-1.pooler.supabase.com:6543/postgres?pgbouncer=true&connection_limit=1"
echo "$CONN3"
echo ""

# Test mit psql (wenn installiert)
echo "🔌 Testing connection with psql (if available)..."
if command -v psql &> /dev/null; then
    echo "Testing Format 1..."
    psql "$CONN1" -c "SELECT 1;" 2>&1 | head -5
    echo ""
else
    echo "psql not installed - skipping connection test"
    echo "Install with: brew install postgresql"
fi

echo ""
echo "✅ Now copy Format 3 into your .env file as:"
echo "SUPABASE_URL=\"$CONN3\""
