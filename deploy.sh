#!/bin/bash

# Clean deploy script for GitHub Pages
# Keeps main branch history clean by deploying only built artifacts

set -e

echo "🚀 Deploying TEP-SLR to GitHub Pages..."

# Ensure we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/site"

# Build the static site
echo "🔨 Building static site..."
npm run build

# Check if dist exists and has content
if [ ! -d "dist" ] || [ -z "$(ls -A dist)" ]; then
    echo "❌ Build failed - dist directory is empty or missing"
    exit 1
fi

echo "📁 Build completed successfully!"

# Temporary directory for gh-pages branch
TEMP_DIR=$(mktemp -d)
echo "📂 Using temporary directory: $TEMP_DIR"

# Clone only the gh-pages branch to temp directory
if git ls-remote --exit-code --heads origin gh-pages >/dev/null 2>&1; then
    echo "📥 Cloning existing gh-pages branch..."
    git clone --depth 1 --branch gh-pages --single-branch "$(git remote get-url origin)" "$TEMP_DIR"
else
    echo "🆕 Creating new gh-pages branch..."
    git clone --depth 1 "$(git remote get-url origin)" "$TEMP_DIR"
    cd "$TEMP_DIR"
    git checkout --orphan gh-pages
    git rm -rf .
    cd - >/dev/null
fi

# Clear existing content (keep .git)
cd "$TEMP_DIR"
find . -maxdepth 1 ! -name '.git' ! -name '.' -exec rm -rf {} +

# Copy built site to temp directory
echo "📋 Copying built site..."
cp -r "$SCRIPT_DIR/site/dist"/* .

# Add .nojekyll to prevent Jekyll processing
touch .nojekyll

# Create a clean commit
git add -A
git config user.name "GitHub Actions Deploy"
git config user.email "deploy@github-actions.local"

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "✅ No changes to deploy"
    rm -rf "$TEMP_DIR"
    exit 0
fi

# Commit with timestamp
COMMIT_MESSAGE="Deploy $(date '+%Y-%m-%d %H:%M:%S UTC')"
git commit -m "$COMMIT_MESSAGE"

# Push to gh-pages
echo "📤 Pushing to gh-pages branch..."
git push origin gh-pages --force

# Cleanup
cd - >/dev/null
rm -rf "$TEMP_DIR"

echo "✅ Deployment complete!"
echo "🌐 Site will be available at: https://$(git remote get-url origin | sed 's/.*github\.com[:/]\([^/]*\)\/\([^.]*\)\.git/\1.github.io\/\2/')/"
