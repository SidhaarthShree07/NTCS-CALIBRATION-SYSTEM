import React from 'react';
import './CameraCard.css';
import SmartThumbnail from './SmartThumbnail';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import API_URL from '../config';

export default function CameraCard({ camera, onDelete }) {
  const navigate = useNavigate();
  const id = camera.id || camera.cameraId;
  const link = camera.link || camera.cameraLink;
  const { location } = camera;
  const placeholder = '/thumbnail-placeholder.svg';
  
  const handleCalibrate = async () => {
    // Pre-cache the video source before navigating
    if (link) {
      try {
        console.log('[CameraCard] Setting video source:', link);
        await axios.post(`${API_URL}/api/set_video_source`, { source: link });
        console.log('[CameraCard] Video source set, background download started');
      } catch (e) {
        console.error('[CameraCard] Failed to set video source:', e);
      }
    }
    
    // Navigate to calibration page with camera context
    navigate('/calibration', { 
      state: { 
        cameraId: id, 
        cameraLink: link, 
        location 
      } 
    });
  };

  return (
    <div className="camera-card">
      <div className="thumb">
        {link ? (
          <SmartThumbnail src={link} alt={`${id} thumbnail`} fallbackSrc={placeholder} />
        ) : (
          <div className="thumb-placeholder"></div>
        )}
        <div className="thumb-overlay">{location}</div>
      </div>
      <div className="card-body">
        <div className="card-title">{id}</div>
        
        <div className="card-actions">
          <button 
            className="btn-calibrate" 
            onClick={handleCalibrate}
            title="Calibrate this camera"
          >
            Calibration ↗
          </button>
          <button 
            className="btn-delete" 
            onClick={() => onDelete && onDelete(camera)}
            title="Delete camera"
          >
            Delete ✕
          </button>
        </div>
      </div>
    </div>
  );
}
