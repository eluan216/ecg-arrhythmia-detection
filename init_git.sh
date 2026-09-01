#!/bin/bash

# Bash script to initialize Git repository and prepare for GitHub

echo ""
echo "================================"
echo "ECG Arrhythmia Detection"
echo "Git Repository Initialization"
echo "================================"
echo ""

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "ERROR: Git not found"
    echo "Please install Git using your package manager or from https://git-scm.com/"
    exit 1
fi

# Check if we're in the project directory
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Not in project root directory"
    exit 1
fi

# Check if .git already exists
if [ -d ".git" ]; then
    echo "Repository already initialized. Skipping git init."
else
    echo "Initializing Git repository..."
    git init
    echo "✓ Repository initialized"
fi

# Check git config
gitname=$(git config user.name)
gitemail=$(git config user.email)

if [ -z "$gitname" ]; then
    echo ""
    echo "Git user not configured. Setting up..."
    read -p "Enter your name: " gitname
    read -p "Enter your email: " gitemail
    git config --global user.name "$gitname"
    git config --global user.email "$gitemail"
    echo "✓ Git user configured"
else
    echo ""
    echo "✓ Git user configured: $gitname <$gitemail>"
fi

# Create initial commit
echo ""
echo "Adding files to Git..."
git add .
echo "✓ Files staged"

echo ""
echo "Creating initial commit..."
git commit -m "Initial project setup: Stage 1 implementation complete"
echo "✓ Commit created"

# Display status
echo ""
echo "================================"
git status

echo ""
echo "================================"
echo "Next Steps:"
echo "================================"
echo "1. Create a repository on GitHub (https://github.com/new)"
echo "2. Add remote:"
echo "   git remote add origin https://github.com/YOUR-USERNAME/ecg-arrhythmia-detection"
echo "3. Push to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "GitHub Actions CI will run on push and verify:"
echo "  - Linting (black, ruff)"
echo "  - Type checking (mypy)"
echo "  - Unit tests (pytest with coverage)"
echo ""
echo "For manual verification:"
echo "  pytest tests/ -v --cov=src"
echo ""
