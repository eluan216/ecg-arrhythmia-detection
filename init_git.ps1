# PowerShell script to initialize Git repository and prepare for GitHub

Write-Host "`n================================" -ForegroundColor Green
Write-Host "ECG Arrhythmia Detection" -ForegroundColor Green
Write-Host "Git Repository Initialization" -ForegroundColor Green
Write-Host "================================`n" -ForegroundColor Green

# Check if git is available
try {
    git --version | Out-Null
} catch {
    Write-Host "ERROR: Git not found in PATH" -ForegroundColor Red
    Write-Host "Please install Git from https://git-scm.com/download/win" -ForegroundColor Red
    exit 1
}

# Check if we're in the project directory
if (-not (Test-Path "pyproject.toml")) {
    Write-Host "ERROR: Not in project root directory" -ForegroundColor Red
    exit 1
}

# Check if .git already exists
if (Test-Path ".git") {
    Write-Host "Repository already initialized. Skipping git init." -ForegroundColor Yellow
} else {
    Write-Host "Initializing Git repository..." -ForegroundColor Cyan
    git init
    Write-Host "✓ Repository initialized" -ForegroundColor Green
}

# Check git config
$email = git config user.email
$name = git config user.name

if (-not $name) {
    Write-Host "`nGit user not configured. Setting up..." -ForegroundColor Yellow
    $gitName = Read-Host "Enter your name"
    $gitEmail = Read-Host "Enter your email"
    git config user.name $gitName
    git config user.email $gitEmail
    Write-Host "✓ Git user configured" -ForegroundColor Green
} else {
    Write-Host "`n✓ Git user configured: $name <$email>" -ForegroundColor Green
}

# Create initial commit
Write-Host "`nAdding files to Git..." -ForegroundColor Cyan
git add .
Write-Host "✓ Files staged" -ForegroundColor Green

Write-Host "`nCreating initial commit..." -ForegroundColor Cyan
git commit -m "Initial project setup: Stage 1 implementation complete"
Write-Host "✓ Commit created" -ForegroundColor Green

# Display status
Write-Host "`n" -ForegroundColor Green
git status

Write-Host "`n================================" -ForegroundColor Green
Write-Host "Next Steps:" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host "1. Create a repository on GitHub (https://github.com/new)"
Write-Host "2. Add remote: git remote add origin https://github.com/YOUR-USERNAME/ecg-arrhythmia-detection"
Write-Host "3. Push to GitHub: git branch -M main; git push -u origin main"
Write-Host ""
Write-Host "GitHub Actions CI will run on push and verify:"
Write-Host "  - Linting (black, ruff)"
Write-Host "  - Type checking (mypy)"
Write-Host "  - Unit tests (pytest with coverage)"
Write-Host ""
Write-Host "For manual verification:"
Write-Host "  pytest tests/ -v --cov=src"
Write-Host "`n"
