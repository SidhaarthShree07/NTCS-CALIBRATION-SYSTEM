import React, { useState } from 'react';
import './CalibrationPanel.css';

function CalibrationPanel({ 
  calibrationData, 
  isAutoCalibrating, 
  onAutoCalibrate, 
  onManualCalibrate,
  onStartSpeed,
  onStopSpeed 
}) {
  const [showManual, setShowManual] = useState(false);
  const [manualData, setManualData] = useState({
    line_A_y: calibrationData.line_A_y || 300,
    line_B_y: calibrationData.line_B_y || 500,
    calib_distance_m: calibrationData.calib_distance_m || 10.0,
    road_width_m: calibrationData.road_width_m || 10.0,
    polygon_x1: 100,
    polygon_y1: 200,
    polygon_x2: 1820,
    polygon_y2: 200,
    polygon_x3: 1820,
    polygon_y3: 1000,
    polygon_x4: 100,
    polygon_y4: 1000
  });

  const handleInputChange = (field, value) => {
    setManualData(prev => ({
      ...prev,
      [field]: parseFloat(value) || 0
    }));
  };

  const handleManualSubmit = (e) => {
    e.preventDefault();
    
    // Build source_points from polygon coordinates
    const source_points = [
      [manualData.polygon_x1, manualData.polygon_y1],
      [manualData.polygon_x2, manualData.polygon_y2],
      [manualData.polygon_x3, manualData.polygon_y3],
      [manualData.polygon_x4, manualData.polygon_y4]
    ];

    onManualCalibrate({
      line_A_y: Math.round(manualData.line_A_y),
      line_B_y: Math.round(manualData.line_B_y),
      calib_distance_m: manualData.calib_distance_m,
      road_width_m: manualData.road_width_m,
      source_points: source_points
    });
  };

  return (
    <div className="calibration-panel">
      <h2>🎛️ Calibration Controls</h2>

      {/* Auto Calibration Section */}
      <div className="control-section">
        <h3>Auto Calibration</h3>
        <button 
          className="btn btn-primary btn-large"
          onClick={onAutoCalibrate}
          disabled={isAutoCalibrating || calibrationData.running}
        >
          {isAutoCalibrating ? (
            <>
              <span className="spinner-small"></span>
              Calibrating...
            </>
          ) : (
            <>🤖 Auto Calibrate (Combined)</>
          )}
        </button>
        <p className="info-text">
          Automatically detects lines using Gemini AI and tracks vehicle for distance estimation
        </p>
      </div>

      {/* Manual Calibration Section */}
      <div className="control-section">
        <h3>Manual Calibration</h3>
        <button 
          className="btn btn-secondary"
          onClick={() => setShowManual(!showManual)}
        >
          {showManual ? '🔼 Hide Manual Controls' : '🔽 Show Manual Controls'}
        </button>

        {showManual && (
          <form onSubmit={handleManualSubmit} className="manual-form">
            <div className="form-section">
              <h4>📏 Calibration Lines</h4>
              <div className="form-group">
                <label>Line A Y-coordinate (Entry/Far):</label>
                <input
                  type="number"
                  value={manualData.line_A_y}
                  onChange={(e) => handleInputChange('line_A_y', e.target.value)}
                  min="0"
                  max="2000"
                />
              </div>
              <div className="form-group">
                <label>Line B Y-coordinate (Exit/Near):</label>
                <input
                  type="number"
                  value={manualData.line_B_y}
                  onChange={(e) => handleInputChange('line_B_y', e.target.value)}
                  min="0"
                  max="2000"
                />
              </div>
            </div>

            <div className="form-section">
              <h4>📐 Distance & Width</h4>
              <div className="form-group">
                <label>Distance between lines (meters):</label>
                <input
                  type="number"
                  step="0.1"
                  value={manualData.calib_distance_m}
                  onChange={(e) => handleInputChange('calib_distance_m', e.target.value)}
                  min="1"
                  max="500"
                />
              </div>
              <div className="form-group">
                <label>Road width (meters):</label>
                <input
                  type="number"
                  step="0.1"
                  value={manualData.road_width_m}
                  onChange={(e) => handleInputChange('road_width_m', e.target.value)}
                  min="3"
                  max="50"
                />
              </div>
            </div>

            <div className="form-section">
              <h4>🔷 Polygon Points (Road Boundary)</h4>
              <div className="polygon-grid">
                <div className="form-group">
                  <label>Point 1 (Top-Left):</label>
                  <div className="coordinate-input">
                    <input
                      type="number"
                      placeholder="X"
                      value={manualData.polygon_x1}
                      onChange={(e) => handleInputChange('polygon_x1', e.target.value)}
                    />
                    <input
                      type="number"
                      placeholder="Y"
                      value={manualData.polygon_y1}
                      onChange={(e) => handleInputChange('polygon_y1', e.target.value)}
                    />
                  </div>
                </div>
                <div className="form-group">
                  <label>Point 2 (Top-Right):</label>
                  <div className="coordinate-input">
                    <input
                      type="number"
                      placeholder="X"
                      value={manualData.polygon_x2}
                      onChange={(e) => handleInputChange('polygon_x2', e.target.value)}
                    />
                    <input
                      type="number"
                      placeholder="Y"
                      value={manualData.polygon_y2}
                      onChange={(e) => handleInputChange('polygon_y2', e.target.value)}
                    />
                  </div>
                </div>
                <div className="form-group">
                  <label>Point 3 (Bottom-Right):</label>
                  <div className="coordinate-input">
                    <input
                      type="number"
                      placeholder="X"
                      value={manualData.polygon_x3}
                      onChange={(e) => handleInputChange('polygon_x3', e.target.value)}
                    />
                    <input
                      type="number"
                      placeholder="Y"
                      value={manualData.polygon_y3}
                      onChange={(e) => handleInputChange('polygon_y3', e.target.value)}
                    />
                  </div>
                </div>
                <div className="form-group">
                  <label>Point 4 (Bottom-Left):</label>
                  <div className="coordinate-input">
                    <input
                      type="number"
                      placeholder="X"
                      value={manualData.polygon_x4}
                      onChange={(e) => handleInputChange('polygon_x4', e.target.value)}
                    />
                    <input
                      type="number"
                      placeholder="Y"
                      value={manualData.polygon_y4}
                      onChange={(e) => handleInputChange('polygon_y4', e.target.value)}
                    />
                  </div>
                </div>
              </div>
            </div>

            <button type="submit" className="btn btn-success btn-large">
              ✅ Apply Manual Calibration
            </button>
          </form>
        )}
      </div>

      {/* Speed Detection Controls */}
      <div className="control-section">
        <h3>Speed Detection</h3>
        <div className="button-group">
          <button 
            className="btn btn-success"
            onClick={onStartSpeed}
            disabled={calibrationData.running}
          >
            ▶️ Start Detection
          </button>
          <button 
            className="btn btn-danger"
            onClick={onStopSpeed}
            disabled={!calibrationData.running}
          >
            ⏹️ Stop Detection
          </button>
        </div>
      </div>
    </div>
  );
}

export default CalibrationPanel;
