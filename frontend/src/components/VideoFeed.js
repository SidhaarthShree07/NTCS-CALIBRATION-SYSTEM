import React from 'react';
import './VideoFeed.css';

function VideoFeed({ endpoint, cameraLink }) {
  const baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:5001';
  
  // Always use backend endpoint for processed feeds
  // The backend now serves the camera stream that was set via /api/set_video_source
  const videoUrl = `${baseUrl}${endpoint}`;

  return (
    <div className="video-feed-container">
      <img 
        src={videoUrl} 
        alt="Video Feed"
        className="video-stream"
        onError={(e) => {
          e.target.style.display = 'none';
          e.target.nextSibling.style.display = 'flex';
        }}
        onLoad={(e) => {
          e.target.style.display = 'block';
          e.target.nextSibling.style.display = 'none';
        }}
      />
      <div className="video-placeholder">
        <div className="spinner"></div>
        <p>{cameraLink ? 'Loading camera stream...' : 'Waiting for video feed...'}</p>
      </div>
    </div>
  );
}

export default VideoFeed;
