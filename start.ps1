# Traffic Speed Detection System - Startup Script
# This script starts both the backend (Flask) and frontend (React)

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Traffic Speed Detection System - Starting...  " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Node.js is installed
Write-Host "[1/4] Checking Node.js installation..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version
    Write-Host "  ✓ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Node.js not found! Please install Node.js from https://nodejs.org/" -ForegroundColor Red
    exit 1
}

# Check if Python is installed
Write-Host "[2/4] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "  ✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Install frontend dependencies if needed
Write-Host "[3/4] Checking frontend dependencies..." -ForegroundColor Yellow
if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "  Installing npm packages (this may take a few minutes)..." -ForegroundColor Yellow
    cd frontend
    npm install
    cd ..
    Write-Host "  ✓ Frontend dependencies installed" -ForegroundColor Green
} else {
    Write-Host "  ✓ Frontend dependencies already installed" -ForegroundColor Green
}

# Start the backend
Write-Host "[4/4] Starting services..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Starting Flask Backend (Port 5001)..." -ForegroundColor Cyan
$backendJob = Start-Job -ScriptBlock {
    Set-Location "d:\Traffic\src"
    python calib_server.py
}

Start-Sleep -Seconds 3

# Start the frontend
Write-Host "Starting React Frontend (Port 3000)..." -ForegroundColor Cyan
$frontendJob = Start-Job -ScriptBlock {
    Set-Location "d:\Traffic\frontend"
    npm start
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "  Services Started Successfully!                " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Backend:  http://localhost:5001" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "The React app will open automatically in your browser." -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop both services..." -ForegroundColor Gray
Write-Host ""

# Wait for user interruption
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    Write-Host ""
    Write-Host "Stopping services..." -ForegroundColor Yellow
    Stop-Job -Job $backendJob
    Stop-Job -Job $frontendJob
    Remove-Job -Job $backendJob
    Remove-Job -Job $frontendJob
    Write-Host "Services stopped." -ForegroundColor Green
}
