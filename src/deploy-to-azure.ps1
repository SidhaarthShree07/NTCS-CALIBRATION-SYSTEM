# Azure Container Apps Deployment Script
# This script builds and deploys the NTCS backend to Azure Container Registry and Container Apps

param(
    [string]$ResourceGroup = "ntcs-rg",
    [string]$Location = "centralindia",
    [string]$RegistryName = "ntcscalibration",
    [string]$ContainerAppName = "ntcs-backend",
    [string]$EnvironmentName = "ntcs-env",
    [string]$ImageTag = "latest"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "NTCS Backend Deployment to Azure" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Login to Azure Container Registry
Write-Host "[1/6] Logging into Azure Container Registry..." -ForegroundColor Yellow
az acr login --name $RegistryName
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to login to ACR" -ForegroundColor Red
    exit 1
}
Write-Host "✓ ACR login successful" -ForegroundColor Green
Write-Host ""

# Step 2: Build Docker Image
Write-Host "[2/6] Building Docker image..." -ForegroundColor Yellow
$imageName = "$RegistryName.azurecr.io/ntcs-backend:$ImageTag"
docker build -t $imageName .
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Docker image built: $imageName" -ForegroundColor Green
Write-Host ""

# Step 3: Push to Azure Container Registry
Write-Host "[3/6] Pushing image to Azure Container Registry..." -ForegroundColor Yellow
docker push $imageName
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to push image to ACR" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Image pushed successfully" -ForegroundColor Green
Write-Host ""

# Step 4: Create or verify Resource Group
Write-Host "[4/6] Verifying resource group..." -ForegroundColor Yellow
az group show --name $ResourceGroup --output none 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Creating resource group: $ResourceGroup" -ForegroundColor Yellow
    az group create --name $ResourceGroup --location $Location
}
Write-Host "✓ Resource group ready" -ForegroundColor Green
Write-Host ""

# Step 5: Create or verify Container Apps Environment
Write-Host "[5/6] Verifying Container Apps environment..." -ForegroundColor Yellow
az containerapp env show --name $EnvironmentName --resource-group $ResourceGroup --output none 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Creating Container Apps environment: $EnvironmentName" -ForegroundColor Yellow
    az containerapp env create `
        --name $EnvironmentName `
        --resource-group $ResourceGroup `
        --location $Location
}
Write-Host "✓ Container Apps environment ready" -ForegroundColor Green
Write-Host ""

# Step 6: Deploy or Update Container App
Write-Host "[6/6] Deploying container app..." -ForegroundColor Yellow

# Get ACR credentials
$acrServer = az acr show --name $RegistryName --query loginServer --output tsv
$acrUsername = az acr credential show --name $RegistryName --query username --output tsv
$acrPassword = az acr credential show --name $RegistryName --query "passwords[0].value" --output tsv

# Check if app exists
az containerapp show --name $ContainerAppName --resource-group $ResourceGroup --output none 2>$null
if ($LASTEXITCODE -ne 0) {
    # Create new container app
    Write-Host "Creating new container app: $ContainerAppName" -ForegroundColor Yellow
    az containerapp create `
        --name $ContainerAppName `
        --resource-group $ResourceGroup `
        --environment $EnvironmentName `
        --image $imageName `
        --target-port 5001 `
        --ingress external `
        --registry-server $acrServer `
        --registry-username $acrUsername `
        --registry-password $acrPassword `
        --cpu 2.0 `
        --memory 4.0Gi `
        --min-replicas 1 `
        --max-replicas 3 `
        --env-vars `
            BACKEND_API_BASE=/api `
            PYTHONUNBUFFERED=1 `
            FORCE_CPU=1
} else {
    # Update existing container app
    Write-Host "Updating existing container app: $ContainerAppName" -ForegroundColor Yellow
    az containerapp update `
        --name $ContainerAppName `
        --resource-group $ResourceGroup `
        --image $imageName
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to deploy container app" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Container app deployed successfully" -ForegroundColor Green
Write-Host ""

# Get the app URL
$appUrl = az containerapp show `
    --name $ContainerAppName `
    --resource-group $ResourceGroup `
    --query properties.configuration.ingress.fqdn `
    --output tsv

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Application URL: https://$appUrl" -ForegroundColor Cyan
Write-Host "Health Check: https://$appUrl/api/status" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Update Vercel environment variable:" -ForegroundColor White
Write-Host "   REACT_APP_API_URL=https://$appUrl" -ForegroundColor White
Write-Host "2. Upload YOLOv8 model (yolov8x.pt) to the container" -ForegroundColor White
Write-Host "3. Set GEMINI_API_KEY environment variable in Azure" -ForegroundColor White
Write-Host ""
