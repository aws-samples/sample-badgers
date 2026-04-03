#!/bin/bash
# Start the BADGERS Local Testing UI
# Runs both the Vite dev server and the Express API server

cd "$(dirname "$0")"

echo "🦡 Starting BADGERS Local Testing..."
echo "   Vite:   http://localhost:5174"
echo "   API:    http://localhost:3457"
echo ""

npm run dev
