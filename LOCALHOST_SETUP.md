# Localhost Installation (Windows)

This guide runs the backend locally on Windows. The frontend is optional and can be started separately.

## Prerequisites
- Python 3.11+
- Git
- FFmpeg in PATH
- (Optional) Docker Desktop

## 1) Clone and create a virtual environment
```powershell
git clone https://github.com/SidhaarthShree07/NTCS-CALIBRATION.git
cd NTCS-CALIBRATION
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2) Install backend dependencies
```powershell
pip install -r src/requirements.txt
```

## 3) Configure environment variables
1. Copy [.env.example](.env.example) to `.env`.
2. Fill in your values (Gemini API key, Azure connection string, and HMAC secret).

Example:
```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=YOUR_ACCOUNT;AccountKey=YOUR_KEY==;EndpointSuffix=core.windows.net
AZURE_CONTAINER_NAME=traffic-violations
HMAC_SECRET=YOUR_STRONG_RANDOM_SECRET
```

## 4) Download model files
Place the following in `src/models/`:
- `yolov8l.pt`
- `yolov8l.onnx`

## 5) Run the backend
```powershell
cd src
python calib_server.py
```

Backend starts at `http://localhost:5001`.

## 6) (Optional) Run the frontend
If you are using the frontend in this repo:
```powershell
cd frontend
npm install
npm start
```

## 7) (Optional) Run with Docker
```powershell
cd src
docker build -t ntcs-backend:local .
docker run -p 5001:5001 --env-file ..\.env ntcs-backend:local
```

## Troubleshooting
- If Gemini features do not work, verify `GEMINI_API_KEY` is set.
- If Azure upload fails, verify `AZURE_STORAGE_CONNECTION_STRING` and `AZURE_CONTAINER_NAME`.
- If FFmpeg is missing, install it and ensure `ffmpeg` is in PATH.
