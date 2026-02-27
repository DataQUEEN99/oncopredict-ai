#!/bin/bash

# OncoPredict AI - GitHub Deployment Helper Script
# This script helps push the code to GitHub

echo "🧬 OncoPredict AI - GitHub Deployment Helper"
echo "=============================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Error: Git repository not initialized"
    echo "Run: git init"
    exit 1
fi

echo "📝 Please provide your GitHub information:"
echo ""

# Get user input
read -p "GitHub Username: " GITHUB_USER
read -p "Repository Name (default: oncopredict-ai): " REPO_NAME
REPO_NAME=${REPO_NAME:-oncopredict-ai}

echo ""
echo "🔧 Configuring Git..."

# Configure git
git config user.name "$GITHUB_USER"
read -p "Git Email: " GIT_EMAIL
git config user.email "$GIT_EMAIL"

echo ""
echo "📦 Repository URL: https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
echo "⚠️  IMPORTANT: Before continuing, please:"
echo "   1. Go to https://github.com/new"
echo "   2. Create a repository named: $REPO_NAME"
echo "   3. Leave it empty (don't initialize with README)"
echo "   4. Generate a Personal Access Token at: https://github.com/settings/tokens"
echo "      - Select 'repo' scope"
echo "      - Copy the token"
echo ""

read -p "Press Enter when ready to continue..."

echo ""
read -sp "GitHub Personal Access Token: " GITHUB_TOKEN
echo ""

# Add remote
echo ""
echo "🔗 Adding GitHub remote..."
git remote remove origin 2>/dev/null
git remote add origin https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git

# Set main branch
echo "🌿 Setting main branch..."
git branch -M main

# Push to GitHub
echo "🚀 Pushing code to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS! Code pushed to GitHub"
    echo ""
    echo "🔗 GitHub Repository: https://github.com/$GITHUB_USER/$REPO_NAME"
    echo ""
    echo "📋 Next Steps:"
    echo "   1. Visit: https://share.streamlit.io/"
    echo "   2. Click 'New app'"
    echo "   3. Select repository: $GITHUB_USER/$REPO_NAME"
    echo "   4. Branch: main"
    echo "   5. Main file: app.py"
    echo "   6. Click 'Deploy!'"
    echo ""
    echo "⏱️  Deployment will take 3-5 minutes"
    echo "🎉 You'll receive a public URL after deployment!"
else
    echo ""
    echo "❌ Error: Failed to push to GitHub"
    echo "Please check:"
    echo "   - Repository exists on GitHub"
    echo "   - Personal Access Token is valid"
    echo "   - Token has 'repo' permissions"
fi
