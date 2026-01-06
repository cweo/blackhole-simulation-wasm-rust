#!/bin/bash
# Build and serve the black hole simulation

set -e

echo "Building WASM..."
~/.cargo/bin/wasm-pack build --target web --release

echo ""
echo "Starting server at http://localhost:8080"
echo "Press Ctrl+C to stop"
echo ""

# Use Python's built-in HTTP server
python3 -m http.server 8080
