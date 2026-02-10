import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import axios from 'axios';
import Header from '../components/Header';
import VideoFeed from '../components/VideoFeed';
import { Sparkles, Calculator, Play, Square } from 'lucide-react';
import './Calibration.css';
import API_URL from '../config';

function Calibration() {
  const location = useLocation();
  const cameraContext = location.state; // { cameraId, cameraLink, location }

  const formatNumber = (value, digits) => {
    const num = Number(value);
    if (!Number.isFinite(num)) return '—';
    return num.toFixed(digits);
  };
  
  const [calibrationData, setCalibrationData] = useState({
    line_A_y: 300,
    line_B_y: 500,
    calib_distance_m: 10.0,
    road_width_m: 10.0,
    source_points: null,
    running: false,
    calibration_method: 'unknown',
    confidence_score: 0.0
  });

  const [isAutoCalibrating, setIsAutoCalibrating] = useState(false);
  const [calibrationStatus, setCalibrationStatus] = useState('');
  const [speedLimit, setSpeedLimit] = useState(60);
  const [showManualForm, setShowManualForm] = useState(false);
  const [manualData, setManualData] = useState({
    line_A_y: 300,
    line_B_y: 500,
    calib_distance_m: 10.0,
    road_width_m: 10.0,
    polygon_x1: 100,
    polygon_y1: 200,
    polygon_x2: 1820,
    polygon_y2: 200,
    polygon_x3: 1820,
    polygon_y3: 1000,
    polygon_x4: 100,
    polygon_y4: 1000
  });

  // Load calibration data from backend
  useEffect(() => {
    const loadCalibration = async () => {
      if (cameraContext && cameraContext.cameraId) {
        try {
          console.log(`[Calibration] Loading data for camera: ${cameraContext.cameraId}`);
          const response = await axios.get(`${API_URL}/api/load_calibration/${cameraContext.cameraId}`);
          
          if (response.data.status === 'success' && response.data.calibration) {
            const cal = response.data.calibration;
            setCalibrationData(prev => ({
              ...prev,
              line_A_y: cal.line_A_y,
              line_B_y: cal.line_B_y,
              calib_distance_m: cal.calib_distance_m,
              road_width_m: cal.road_width_m,
              source_points: cal.source_points,
              calibration_method: cal.method,
              confidence_score: cal.confidence
            }));
            
            if (response.data.speed_limit) {
              setSpeedLimit(response.data.speed_limit);
            }
            
            if (response.data.auto_started) {
              setCalibrationStatus('Calibration loaded from backend');
            } else {
              setCalibrationStatus('Calibration loaded from backend');
            }
          }
        } catch (error) {
          console.error('[Calibration] Error loading calibration:', error);
          setCalibrationStatus('Using default calibration values');
        }
      }
    };
    
    loadCalibration();
  }, [cameraContext]);

  // Fetch current status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await axios.get(`${API_URL}/api/status`);
        setCalibrationData(prev => ({
          ...prev,
          running: response.data.running
        }));
      } catch (error) {
        console.error('Error fetching status:', error);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  // Set video source
  useEffect(() => {
    if (cameraContext && cameraContext.cameraLink) {
      axios.post(`${API_URL}/api/set_video_source`, { source: cameraContext.cameraLink })
        .then(res => {
          console.log('Video source set:', res.data);
        })
        .catch(err => {
          console.error('Failed to set video source:', err);
        });
    }
  }, [cameraContext]);

  const handleAutoCalibration = async () => {
    setIsAutoCalibrating(true);
    setCalibrationStatus('Starting auto calibration... (This may take up to 2 minutes)');

    try {
      // Increase timeout for CPU-intensive segmentation + Gemini API
      const response = await axios.post(`${API_URL}/api/auto_calibrate_full`, {}, {
        timeout: 300000 // 5 minutes
      });
      
      if (response.data.status === 'success') {
        setCalibrationStatus('Auto calibration complete!');
        setCalibrationData(prev => ({
          ...prev,
          line_A_y: response.data.line_A_y,
          line_B_y: response.data.line_B_y,
          calib_distance_m: response.data.distance_m,
          road_width_m: response.data.road_width_m,
          source_points: response.data.source_points,
          calibration_method: 'auto_combined',
          confidence_score: response.data.confidence || 0.8
        }));
      } else {
        setCalibrationStatus(`Auto calibration failed: ${response.data.message}`);
      }
    } catch (error) {
      if (error.code === 'ECONNABORTED') {
        setCalibrationStatus('Calibration is still running on the backend. Waiting for results...');
        // Poll once to refresh values if backend completed after timeout
        try {
          if (cameraContext && cameraContext.cameraId) {
            const res = await axios.get(`${API_URL}/api/load_calibration/${cameraContext.cameraId}`);
            if (res.data.status === 'success' && res.data.calibration) {
              const cal = res.data.calibration;
              setCalibrationData(prev => ({
                ...prev,
                line_A_y: cal.line_A_y,
                line_B_y: cal.line_B_y,
                calib_distance_m: cal.calib_distance_m,
                road_width_m: cal.road_width_m,
                source_points: cal.source_points,
                calibration_method: cal.method,
                confidence_score: cal.confidence
              }));
              if (res.data.speed_limit) {
                setSpeedLimit(res.data.speed_limit);
              }
              setCalibrationStatus('Calibration completed on backend.');
            }
          }
        } catch (reloadErr) {
          setCalibrationStatus('Error: Request timeout. Check backend logs and try again.');
        }
      } else {
        setCalibrationStatus(`Error: ${error.response?.data?.message || error.message}`);
      }
    } finally {
      setIsAutoCalibrating(false);
    }
  };

  const handleManualCalibration = async (e) => {
    e.preventDefault();
    
    const source_points = [
      [manualData.polygon_x1, manualData.polygon_y1],
      [manualData.polygon_x2, manualData.polygon_y2],
      [manualData.polygon_x3, manualData.polygon_y3],
      [manualData.polygon_x4, manualData.polygon_y4]
    ];

    try {
      const response = await axios.post(`${API_URL}/api/manual_calibrate`, {
        line_A_y: Math.round(manualData.line_A_y),
        line_B_y: Math.round(manualData.line_B_y),
        calib_distance_m: manualData.calib_distance_m,
        road_width_m: manualData.road_width_m,
        source_points: source_points
      });
      
      if (response.data.status === 'success') {
        setCalibrationStatus('Manual calibration applied successfully!');
        setCalibrationData(prev => ({
          ...prev,
          line_A_y: manualData.line_A_y,
          line_B_y: manualData.line_B_y,
          calib_distance_m: manualData.calib_distance_m,
          road_width_m: manualData.road_width_m,
          source_points: source_points,
          calibration_method: 'manual',
          confidence_score: 1.0
        }));
        setShowManualForm(false);
      } else {
        setCalibrationStatus(`Manual calibration failed: ${response.data.message}`);
      }
    } catch (error) {
      setCalibrationStatus(`Error: ${error.response?.data?.message || error.message}`);
    }
  };

  const handleStartSpeed = async () => {
    try {
      const response = await axios.post(`${API_URL}/api/start_speed`);
      if (response.data.status === 'success') {
        setCalibrationStatus('Speed detection started!');
        setCalibrationData(prev => ({ ...prev, running: true }));
      }
    } catch (error) {
      setCalibrationStatus(`Error starting speed detection: ${error.message}`);
    }
  };

  const handleStopSpeed = async () => {
    try {
      const response = await axios.post(`${API_URL}/api/stop_speed`);
      if (response.data.status === 'success') {
        setCalibrationStatus('Speed detection stopped');
        setCalibrationData(prev => ({ ...prev, running: false }));
      }
    } catch (error) {
      setCalibrationStatus(`Error stopping speed detection: ${error.message}`);
    }
  };

  const handleInputChange = (field, value) => {
    setManualData(prev => ({
      ...prev,
      [field]: parseFloat(value) || 0
    }));
  };

  return (
    <>
      <Header />
      <div className="calibration-page">
        <div className="calibration-container">
          {/* Left Column - Controls */}
          <div className="calibration-left">
            {/* Calibration Controls Card */}
            <div className="control-card">
              <div className="card-header">
                <Calculator size={20} />
                <h2>Calibration Controls</h2>
              </div>

              {/* Auto Calibration Section */}
              <div className="control-section auto-calib-section">
                <div className="section-header">
                  <Sparkles size={16} />
                  <span>Auto Calibration</span>
                </div>
                <button 
                  className="btn-auto-calibrate"
                  onClick={handleAutoCalibration}
                  disabled={isAutoCalibrating || calibrationData.running}
                >
                  {isAutoCalibrating ? (
                    <>
                      <div className="spinner-small"></div>
                      Calibrating...
                    </>
                  ) : (
                    <>
                      Calibration
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M7 17L17 7M17 7H7M17 7V17"/>
                      </svg>
                    </>
                  )}
                </button>
                <p className="calibrate-description">
                  Automatically detects lines using Gemini AI and tracks vehicle for distance estimation
                </p>
              </div>

              {/* Manual Calibration Section */}
              <div className="control-section manual-calib-section">
                <div className="section-header">
                  <Calculator size={16} />
                  <span>Manual Calibration</span>
                </div>
                <button 
                  className="btn-manual"
                  onClick={() => setShowManualForm(!showManualForm)}
                >
                  Calibration
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    {showManualForm ? (
                      <path d="M18 15l-6-6-6 6"/>
                    ) : (
                      <path d="M6 9l6 6 6-6"/>
                    )}
                  </svg>
                </button>

                {showManualForm && (
                  <form onSubmit={handleManualCalibration} className="manual-form">
                    <div className="form-fields">
                      <div className="form-group">
                        <label>Line A Y (Entry/Far)</label>
                        <input
                          type="number"
                          value={manualData.line_A_y}
                          onChange={(e) => handleInputChange('line_A_y', e.target.value)}
                        />
                      </div>
                      <div className="form-group">
                        <label>Line B Y (Exit/Near)</label>
                        <input
                          type="number"
                          value={manualData.line_B_y}
                          onChange={(e) => handleInputChange('line_B_y', e.target.value)}
                        />
                      </div>
                      <div className="form-group">
                        <label>Distance (meters)</label>
                        <input
                          type="number"
                          step="0.1"
                          value={manualData.calib_distance_m}
                          onChange={(e) => handleInputChange('calib_distance_m', e.target.value)}
                        />
                      </div>
                      <div className="form-group">
                        <label>Road Width (meters)</label>
                        <input
                          type="number"
                          step="0.1"
                          value={manualData.road_width_m}
                          onChange={(e) => handleInputChange('road_width_m', e.target.value)}
                        />
                      </div>
                    </div>
                    <button type="submit" className="btn-submit">
                      Apply Calibration
                    </button>
                  </form>
                )}
              </div>
            </div>

            {/* Speed Detection Card */}
            <div className="control-card speed-card">
              <div className="card-header">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
                </svg>
                <h2>Speed Detection</h2>
              </div>

              <div className="detection-status">
                <div className="status-indicator">
                  <span className="status-label">Detection</span>
                  <span className={`status-badge ${calibrationData.running ? 'running' : 'stopped'}`}>
                    <span className="status-dot"></span>
                    {calibrationData.running ? 'Running' : 'Stopped'}
                  </span>
                </div>
              </div>

              <div className="speed-controls">
                <button 
                  className="btn-start"
                  onClick={handleStartSpeed}
                  disabled={calibrationData.running}
                >
                  <Play size={16} />
                  Start
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M7 17L17 7M17 7H7M17 7V17"/>
                  </svg>
                </button>
                <button 
                  className="btn-stop"
                  onClick={handleStopSpeed}
                  disabled={!calibrationData.running}
                >
                  <Square size={16} />
                  Stop
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M6 18L18 6M6 6l12 12"/>
                  </svg>
                </button>
              </div>

              <div className="speed-limit-box">
                <span className="limit-label">Speed Limit:</span>
                <span className="limit-value">{speedLimit} Km/h</span>
              </div>
            </div>
          </div>

          {/* Center Column - Video Feed */}
          <div className="calibration-center">
            <div className="video-card">
              <div className="video-header">
                <span className="feed-label">
                  {calibrationData.running ? 'Speed Detection Feed' : 'Original Feed'}
                </span>
                <span className={`feed-status ${calibrationData.running ? 'active' : 'inactive'}`}>
                  <span className="status-dot"></span>
                  {calibrationData.running ? 'Active' : 'Inactive'}
                </span>
              </div>
              <div className="video-wrapper">
                <VideoFeed 
                  endpoint={calibrationData.running ? '/speed_feed' : '/video_feed'} 
                  cameraLink={!calibrationData.running ? cameraContext?.cameraLink : undefined}
                />
              </div>
            </div>
          </div>

          {/* Right Column - System Status */}
          <div className="calibration-right">
            <div className="status-card">
              <div className="card-header">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="3" width="7" height="7"/>
                  <rect x="14" y="3" width="7" height="7"/>
                  <rect x="14" y="14" width="7" height="7"/>
                  <rect x="3" y="14" width="7" height="7"/>
                </svg>
                <h2>System Status</h2>
              </div>

              <div className="status-content">
                <div className="status-message">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
                  </svg>
                  {calibrationStatus || 'Ready'}
                </div>

                {/* Calibration Stats */}
                <div className="stats-section">
                  <h3>Calibration Stats</h3>
                  <div className="stats-grid">
                    <div className="stat-item">
                      <span className="stat-label">Method</span>
                      <span className="stat-value stat-value--method">{calibrationData.calibration_method || 'None'}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Confidence</span>
                      <span className="stat-value">{formatNumber(calibrationData.confidence_score * 100, 0)}%</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Line A (Y)</span>
                      <span className="stat-value">{calibrationData.line_A_y}px</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Line B (Y)</span>
                      <span className="stat-value">{calibrationData.line_B_y}px</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Distance</span>
                      <span className="stat-value">{formatNumber(calibrationData.calib_distance_m, 1)}m</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Road Width</span>
                      <span className="stat-value">{formatNumber(calibrationData.road_width_m, 1)}m</span>
                    </div>
                  </div>
                </div>

                {/* Polygon Points */}
                {calibrationData.source_points && (
                  <div className="polygon-section">
                    <h3>Polygon Points</h3>
                    <div className="polygon-grid">
                      {calibrationData.source_points.map((point, index) => (
                        <div key={index} className="polygon-point">
                          <span className="point-label">P{index + 1}</span>
                          <span className="point-coords">
                            ({Math.round(point[0])}, {Math.round(point[1])})
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default Calibration;
