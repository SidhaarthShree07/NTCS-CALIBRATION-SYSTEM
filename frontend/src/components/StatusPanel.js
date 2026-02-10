import React from 'react';
import './StatusPanel.css';

function StatusPanel({ calibrationData, calibrationStatus, speedStats, speedLimit }) {
  const getStatusColor = () => {
    if (calibrationStatus.includes('✅')) return '#38ef7d';
    if (calibrationStatus.includes('❌')) return '#f45c43';
    return '#667eea';
  };

  return (
    <div className="status-panel">
      <h2>📊 System Status</h2>

      {/* Calibration Status Message */}
      {calibrationStatus && (
        <div className="status-message" style={{ borderLeftColor: getStatusColor() }}>
          {calibrationStatus}
        </div>
      )}

      {/* Speed Limit Display */}
      {speedLimit !== null && speedLimit !== undefined && (
        <div className="status-section">
          <h3>⚡ Speed Limit</h3>
          <div className="speed-limit-display">
            <div className="speed-limit-value">{speedLimit} km/h</div>
          </div>
        </div>
      )}

      {/* Current Calibration Values */}
      <div className="status-section">
        <h3>Current Calibration</h3>
        <div className="status-grid">
          <div className="status-item">
            <span className="status-label">Method:</span>
            <span className="status-value">{calibrationData.calibration_method || 'None'}</span>
          </div>
          <div className="status-item">
            <span className="status-label">Confidence:</span>
            <span className="status-value">
              {(calibrationData.confidence_score * 100).toFixed(0)}%
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Line A (Y):</span>
            <span className="status-value">{calibrationData.line_A_y}px</span>
          </div>
          <div className="status-item">
            <span className="status-label">Line B (Y):</span>
            <span className="status-value">{calibrationData.line_B_y}px</span>
          </div>
          <div className="status-item">
            <span className="status-label">Distance:</span>
            <span className="status-value">{calibrationData.calib_distance_m.toFixed(1)}m</span>
          </div>
          <div className="status-item">
            <span className="status-label">Road Width:</span>
            <span className="status-value">{calibrationData.road_width_m.toFixed(1)}m</span>
          </div>
        </div>
      </div>

      {/* Polygon Points */}
      {calibrationData.source_points && (
        <div className="status-section">
          <h3>Polygon Points</h3>
          <div className="polygon-display">
            {calibrationData.source_points.map((point, index) => (
              <div key={index} className="point-item">
                <span className="point-label">P{index + 1}:</span>
                <span className="point-coords">({point[0].toFixed(0)}, {point[1].toFixed(0)})</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Detection Status */}
      <div className="status-section">
        <h3>Detection Status</h3>
        <div className="detection-status">
          <div className={`status-indicator ${calibrationData.running ? 'active' : 'inactive'}`}>
            {calibrationData.running ? (
              <>
                <span className="pulse"></span>
                <span>🟢 Running</span>
              </>
            ) : (
              <span>⚫ Stopped</span>
            )}
          </div>
        </div>
      </div>

      {/* Speed Statistics */}
      {speedStats.length > 0 && (
        <div className="status-section">
          <h3>Recent Detections</h3>
          <div className="speed-stats">
            {speedStats.slice(0, 5).map((stat, index) => (
              <div key={index} className="speed-item">
                <span className="speed-vehicle">{stat.vehicle_type}</span>
                <span className="speed-value">{stat.speed.toFixed(1)} km/h</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default StatusPanel;
