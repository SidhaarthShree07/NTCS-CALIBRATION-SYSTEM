# Traffic Speed Detection System - React Frontend

## Overview
This is a modern React-based frontend for the Traffic Speed Detection System with side-by-side video feeds, combined auto calibration, and manual calibration controls.

## Features

### 🎯 Main Features
- **Side-by-Side Video Feeds**: Original feed and speed detection feed displayed simultaneously
- **Combined Auto Calibration**: Single button that runs both Gemini line detection and tracked vehicle calibration sequentially
- **Manual Calibration**: Complete form to manually set all calibration parameters:
  - Line A and Line B Y-coordinates
  - Distance between lines (meters)
  - Road width (meters)
  - Polygon points (4 coordinates for road boundary)
- **Real-time Status**: Live display of calibration values, detection status, and statistics
- **Responsive Design**: Works on desktop and tablet devices

## Installation

### Backend Setup

1. Install Python dependencies:
```bash
cd d:\Traffic
pip install -r requirements.txt
```

2. Start the Flask backend:
```bash
cd d:\Traffic\src
python calib_server.py
```

The backend will run on `http://localhost:5001`

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd d:\Traffic\frontend
npm install
```

2. Start the React development server:
```bash
npm start
```

The frontend will open automatically at `http://localhost:3000`

## Usage

### Combined Auto Calibration
1. Click **"🤖 Auto Calibrate (Combined)"** button
2. The system will:
   - First: Use Gemini AI to detect optimal calibration lines and road width
   - Second: Track a vehicle between lines to estimate distance
   - Result: Complete calibration with high confidence

### Manual Calibration
1. Click **"🔽 Show Manual Controls"** to expand the manual form
2. Enter values:
   - **Line A Y-coordinate**: Y-pixel position of entry line (far/top)
   - **Line B Y-coordinate**: Y-pixel position of exit line (near/bottom)
   - **Distance between lines**: Real-world distance in meters
   - **Road width**: Real-world width of road in meters
   - **Polygon Points**: 4 corner points defining the road boundary trapezoid
     - Point 1: Top-Left (X, Y)
     - Point 2: Top-Right (X, Y)
     - Point 3: Bottom-Right (X, Y)
     - Point 4: Bottom-Left (X, Y)
3. Click **"✅ Apply Manual Calibration"**

### Speed Detection
1. After calibration (auto or manual), click **"▶️ Start Detection"**
2. Watch real-time speed measurements in both video feeds
3. Click **"⏹️ Stop Detection"** when done

## API Endpoints

### Backend Endpoints (Flask - Port 5001)

- `GET /api/status` - Get current calibration status and values
- `POST /api/auto_calibrate_full` - Run combined auto calibration
- `POST /api/manual_calibrate` - Apply manual calibration values
- `POST /api/start_speed` - Start speed detection
- `POST /api/stop_speed` - Stop speed detection
- `GET /video_feed` - Original video feed (MJPEG stream)
- `GET /speed_frame` - Speed detection feed (MJPEG stream)

### Manual Calibration Request Format
```json
{
  "line_A_y": 300,
  "line_B_y": 500,
  "calib_distance_m": 10.0,
  "road_width_m": 10.0,
  "source_points": [
    [100, 200],
    [1820, 200],
    [1820, 1000],
    [100, 1000]
  ]
}
```

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── VideoFeed.js          # Video stream component
│   │   ├── VideoFeed.css
│   │   ├── CalibrationPanel.js   # Auto + Manual calibration controls
│   │   ├── CalibrationPanel.css
│   │   ├── StatusPanel.js        # Status display and statistics
│   │   └── StatusPanel.css
│   ├── App.js                    # Main application component
│   ├── App.css
│   ├── index.js
│   └── index.css
├── package.json
└── README.md
```

## Technology Stack

### Frontend
- **React 18** - UI framework
- **Axios** - HTTP client for API calls
- **CSS Grid & Flexbox** - Responsive layout

### Backend
- **Flask** - Web framework
- **Flask-CORS** - Cross-origin support
- **OpenCV** - Video processing
- **YOLOv8** - Object detection
- **Gemini AI** - Line detection and calibration
- **Supervision** - Tracking library

## Tips

### Getting Best Calibration Results

**Auto Calibration:**
- Ensure good lighting and clear road markings
- Let the system track a vehicle completely from Line A to Line B
- Works best with highway footage with visible lane markings

**Manual Calibration:**
- Use a measuring tool or reference to determine real distances
- Polygon points should form a trapezoid (narrower at top, wider at bottom) for perspective correction
- Line A should be in upper 15-25% of polygon, Line B in lower 70-80%
- Typical road widths:
  - Highway: 10-12 meters (2-3 lanes)
  - Urban road: 6-8 meters (2 lanes)
  - Single lane: 3-4 meters

### Troubleshooting

**Video feeds not showing:**
- Check that backend is running on port 5001
- Verify video file path in `calib_server.py` (VIDEO_PATH variable)
- Check browser console for errors

**Auto calibration failing:**
- Ensure Gemini API key is configured
- Check that video has clear road markings
- Verify sufficient vehicle traffic for tracking

**Speed values seem incorrect:**
- Verify calibration distance is accurate
- Check that trapezoid polygon matches road perspective
- Ensure road width is realistic

## Development

### Running in Development Mode

Backend (with auto-reload disabled):
```bash
cd d:\Traffic\src
python calib_server.py
```

Frontend (with hot-reload):
```bash
cd d:\Traffic\frontend
npm start
```

### Building for Production

```bash
cd d:\Traffic\frontend
npm run build
```

The optimized production build will be in `frontend/build/`

## License

See main project LICENSE file.
