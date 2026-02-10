# Dependencies & Docker Configuration - Summary

## ✅ Fixed Issues

### 1. **Missing Dependencies Added to requirements.txt**

**Previously Missing:**
- ❌ `torch` - Required for CUDA/PyTorch operations
- ❌ `torchvision` - Required for image transformations  
- ❌ `scikit-learn` - Required for RANSAC regression in track.py
- ❌ `langchain-core` - Required for langchain message types

**Now Included:**
- ✅ `torch>=2.0.0`
- ✅ `torchvision>=0.15.0`
- ✅ `scikit-learn`
- ✅ `langchain-core`

### 2. **Dockerfile Improvements**

**Added:**
- ✅ `wget` system package for downloading files
- ✅ CPU-only PyTorch installation (smaller image, faster builds)
- ✅ `FORCE_CPU=1` environment variable
- ✅ Separate GPU Dockerfile (`Dockerfile.gpu`)

## 📦 Complete Dependency List

### Core ML & CV
- `torch>=2.0.0` - PyTorch for deep learning
- `torchvision>=0.15.0` - Vision models and transforms
- `ultralytics` - YOLOv8 detection
- `opencv-contrib-python>=4.8.0` - Computer vision

### Tracking
- `deep-sort-realtime` - Object tracking
- `supervision` - Detection utilities

### Scientific
- `numpy` - Array operations
- `scipy` - Scientific computing
- `scikit-learn` - RANSAC, ML utilities
- `scikit-image` - Image processing

### Web Framework
- `flask` - Web server
- `flask-cors` - CORS support

### AI/LLM
- `google-generativeai` - Gemini API
- `langchain` - LLM framework
- `langchain-google-genai` - Gemini integration
- `langchain-core` - Core langchain types

### Cloud
- `azure-storage-blob>=12.19.0` - Azure blob storage

### Utilities
- `pyyaml` - YAML parsing
- `Pillow` - Image processing
- `matplotlib` - Plotting
- `requests` - HTTP client
- `python-dotenv` - Environment variables

## 🐳 Dockerfile Comparison

### Standard Dockerfile (CPU-only)
**Use for:** Azure Container Apps, most cloud deployments
- Base: `python:3.11-slim`
- PyTorch: CPU-only version (~300MB smaller)
- Build time: ~5-10 minutes
- Image size: ~2-3GB
- No GPU required

### Dockerfile.gpu (CUDA-enabled)
**Use for:** Azure VM with NVIDIA GPU, dedicated servers
- Base: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
- PyTorch: CUDA 11.8 version
- Build time: ~10-15 minutes
- Image size: ~5-7GB
- Requires NVIDIA GPU

## 🚀 Deployment Options

### Option 1: Azure Container Apps (Recommended for MVP)
```bash
# Uses standard Dockerfile (CPU-only)
docker build -t ntcs-backend .
```
**Pros:**
- ✅ Auto-scaling
- ✅ Pay-per-use pricing
- ✅ Easy deployment
- ✅ Managed infrastructure

**Cons:**
- ❌ No GPU support
- ❌ Slower inference (~2-3 FPS)

**Estimated Cost:** $30-50/month

### Option 2: Azure VM with GPU
```bash
# Uses Dockerfile.gpu (CUDA-enabled)
docker build -f Dockerfile.gpu -t ntcs-backend-gpu .
```
**Pros:**
- ✅ GPU acceleration
- ✅ Fast inference (~15-30 FPS)
- ✅ Better performance

**Cons:**
- ❌ More expensive
- ❌ Manual scaling
- ❌ More complex setup

**Estimated Cost:** $200-400/month (NC6 Standard with NVIDIA K80)

## 🔧 Build Instructions

### Local Build (CPU)
```powershell
cd D:\Traffic\src
docker build -t ntcs-backend:latest .
```

### Local Build (GPU)
```powershell
cd D:\Traffic\src
docker build -f Dockerfile.gpu -t ntcs-backend-gpu:latest .
```

### Test Locally
```powershell
# CPU version
docker run -p 5001:5001 `
  -e BACKEND_API_BASE=https://nextgen-fv1h.onrender.com/api `
  -e GEMINI_API_KEY=your-key `
  -v ${PWD}/temp:/app/temp `
  -v ${PWD}/models:/app/models `
  ntcs-backend:latest

# GPU version (requires NVIDIA Docker runtime)
docker run --gpus all -p 5001:5001 `
  -e BACKEND_API_BASE=https://nextgen-fv1h.onrender.com/api `
  -e GEMINI_API_KEY=your-key `
  -v ${PWD}/temp:/app/temp `
  -v ${PWD}/models:/app/models `
  ntcs-backend-gpu:latest
```

## ⚠️ Important Notes

1. **Model File**: YOLOv8x model (~130MB) should be mounted as volume or uploaded to Azure File Share

2. **CUDA in Code**: The code has fallback logic to use CPU if CUDA is not available:
   ```python
   device = 0 if torch.cuda.is_available() else 'cpu'
   ```

3. **Memory Requirements**:
   - CPU-only: 4GB RAM minimum
   - GPU: 8GB VRAM (RTX 4070 tested)

4. **Performance**:
   - CPU: ~2-3 FPS processing
   - GPU: ~15-30 FPS processing

## 🎯 Recommendation

**For MVP/Testing:**
- Use standard `Dockerfile` (CPU)
- Deploy to Azure Container Apps
- Lower cost, easier setup

**For Production:**
- Use `Dockerfile.gpu` 
- Deploy to Azure VM with GPU
- Better performance for real-time processing

## 📝 Next Steps

1. ✅ Dependencies fixed
2. ✅ Dockerfile optimized
3. ⏳ Test Docker build locally
4. ⏳ Deploy to Azure
5. ⏳ Configure Vercel with backend URL
