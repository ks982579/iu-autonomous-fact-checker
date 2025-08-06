#!/bin/bash
# Simple shell script version of extension builder

set -e  # Exit on any error

echo "🏗️  Chrome Extension Build Script"
echo "=================================================="

# Check if config.json exists
if [ ! -f "config.json" ]; then
    echo "❌ Error: config.json not found in root directory"
    exit 1
fi

echo "📋 Found config.json"

# Create extension public directory if it doesn't exist
mkdir -p extensions/my-chrome-ext/public

# Copy config to extension directory
echo "🔄 Syncing configuration..."
cp config.json extensions/my-chrome-ext/public/config.json
echo "✅ Config synced: config.json → extensions/my-chrome-ext/public/config.json"

# Check if bun is available
if ! command -v bun &> /dev/null; then
    echo "❌ Error: 'bun' command not found. Make sure bun is installed and in PATH"
    exit 1
fi

# Change to extension directory and build
echo "🔨 Building extension..."
cd extensions/my-chrome-ext

if bun run build:extension; then
    echo "✅ Extension build completed successfully!"
else
    echo "❌ Extension build failed!"
    exit 1
fi

# Go back to root
cd ../../

# Verify key files exist
echo "🔍 Verifying build output..."
DIST_DIR="extensions/my-chrome-ext/dist"

if [ -f "$DIST_DIR/manifest.json" ] && [ -f "$DIST_DIR/content.js" ] && [ -f "$DIST_DIR/config.json" ]; then
    echo "✅ Build verification passed!"
else
    echo "❌ Build verification failed - missing required files"
    exit 1
fi

echo ""
echo "🚀 Next Steps:"
echo "1. Load extension in Chrome from: $(pwd)/$DIST_DIR"
echo "2. Start API with: python run_api.py"
echo "3. Test by clicking extension icon → YES"
echo ""
echo "✅ Extension build completed successfully!"