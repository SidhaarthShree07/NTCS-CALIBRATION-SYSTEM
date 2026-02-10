# Azure Container Apps Deployment Guide

## Prerequisites

1. **Azure CLI** installed: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli
2. **Docker** installed and running
3. **Azure subscription** active

## Step 1: Login to Azure

```bash
az login
```

## Step 2: Set Variables

```bash
# Set your variables
RESOURCE_GROUP="ntcs-traffic-rg"
LOCATION="eastus"
CONTAINER_REGISTRY="ntcstrafficregistry"
CONTAINER_APP_ENV="ntcs-traffic-env"
CONTAINER_APP_NAME="ntcs-backend"
IMAGE_NAME="ntcs-traffic-backend"
```

## Step 3: Create Resource Group

```bash
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION
```

## Step 4: Create Azure Container Registry (ACR)

```bash
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $CONTAINER_REGISTRY \
  --sku Basic \
  --admin-enabled true
```

## Step 5: Build and Push Docker Image to ACR

```bash
# Login to ACR
az acr login --name $CONTAINER_REGISTRY

# Build the image
docker build -t $IMAGE_NAME:latest .

# Tag the image
docker tag $IMAGE_NAME:latest $CONTAINER_REGISTRY.azurecr.io/$IMAGE_NAME:latest

# Push to ACR
docker push $CONTAINER_REGISTRY.azurecr.io/$IMAGE_NAME:latest
```

## Step 6: Create Container Apps Environment

```bash
az containerapp env create \
  --name $CONTAINER_APP_ENV \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION
```

## Step 7: Deploy Container App

```bash
# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $CONTAINER_REGISTRY --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name $CONTAINER_REGISTRY --query passwords[0].value --output tsv)

# Create container app
az containerapp create \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $CONTAINER_APP_ENV \
  --image $CONTAINER_REGISTRY.azurecr.io/$IMAGE_NAME:latest \
  --registry-server $CONTAINER_REGISTRY.azurecr.io \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --target-port 5001 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --cpu 2.0 \
  --memory 4.0Gi \
  --env-vars \
    BACKEND_API_BASE=[YOUR BACKEND] \
    GEMINI_API_KEY=[YOUR KEY]
```

## Step 8: Get Application URL

```bash
az containerapp show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn \
  --output tsv
```

The URL will be something like: `https://ntcs-backend.something.eastus.azurecontainerapps.io`

## Step 9: Upload YOLOv8 Model

Since the model file is large (~130MB), you have two options:

### Option A: Include in Docker Image (Slower builds)
Uncomment the model copy line in Dockerfile and rebuild.

### Option B: Azure File Share (Recommended)

1. Create storage account:
```bash
STORAGE_ACCOUNT="ntcstrafficstorage"
FILE_SHARE="models"

az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS

az storage share create \
  --name $FILE_SHARE \
  --account-name $STORAGE_ACCOUNT
```

2. Upload model:
```bash
az storage file upload \
  --account-name $STORAGE_ACCOUNT \
  --share-name $FILE_SHARE \
  --source ./models/yolov8x.pt \
  --path yolov8x.pt
```

3. Mount to container app:
```bash
STORAGE_KEY=$(az storage account keys list \
  --account-name $STORAGE_ACCOUNT \
  --query [0].value \
  --output tsv)

az containerapp update \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --set-env-vars STORAGE_ACCOUNT=$STORAGE_ACCOUNT STORAGE_KEY=$STORAGE_KEY
```

## Step 10: Update Vercel Frontend

Go to Vercel dashboard and update environment variable:
```
REACT_APP_API_URL=https://[your-container-app-url]
```

## Monitoring and Logs

View live logs:
```bash
az containerapp logs show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --follow
```

## Updating the App

When you make code changes:

```bash
# Rebuild image
docker build -t $IMAGE_NAME:latest .

# Tag with new version
docker tag $IMAGE_NAME:latest $CONTAINER_REGISTRY.azurecr.io/$IMAGE_NAME:v2

# Push to ACR
docker push $CONTAINER_REGISTRY.azurecr.io/$IMAGE_NAME:v2

# Update container app
az containerapp update \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --image $CONTAINER_REGISTRY.azurecr.io/$IMAGE_NAME:v2
```

## Cost Optimization

Container Apps pricing is based on:
- vCPU seconds
- Memory GB seconds
- HTTP requests

Estimated cost with current config: ~$30-50/month

To reduce costs:
```bash
# Scale down when not in use
az containerapp update \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --min-replicas 0 \
  --max-replicas 1
```

## Troubleshooting

1. **Container fails to start:**
   ```bash
   az containerapp logs show \
     --name $CONTAINER_APP_NAME \
     --resource-group $RESOURCE_GROUP \
     --tail 100
   ```

2. **Out of memory:**
   Increase memory allocation:
   ```bash
   az containerapp update \
     --name $CONTAINER_APP_NAME \
     --resource-group $RESOURCE_GROUP \
     --memory 6.0Gi
   ```

3. **CORS issues:**
   Update CORS in `calib_server.py` to allow your Vercel domain.

## Cleanup (Delete everything)

```bash
az group delete --name $RESOURCE_GROUP --yes --no-wait
```

## Alternative: Azure App Service (Simpler but more expensive)

If you prefer simpler deployment (no Docker):

```bash
az webapp up \
  --name ntcs-traffic-backend \
  --resource-group $RESOURCE_GROUP \
  --runtime "PYTHON:3.11" \
  --sku B2
```

Note: Requires `startup.txt` with: `gunicorn --bind=0.0.0.0:8000 --timeout 600 calib_server:app`
