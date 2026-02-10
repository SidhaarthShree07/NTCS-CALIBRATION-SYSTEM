import React, { useEffect, useState } from 'react';
import AddCameraModal from '../components/AddCameraModal';
import DeleteCameraModal from '../components/DeleteCameraModal';
import CameraCard from '../components/CameraCard';
import Header from '../components/Header';
import './Home.css';
import MaintenancePage from './MaintenancePage';
import axios from 'axios';
import API_URL from '../config';

export default function Home() {
  const [modalOpen, setModalOpen] = useState(false);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [cameraToDelete, setCameraToDelete] = useState(null);
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [maintenance, setMaintenance] = useState(false);
  const [videoLoading, setVideoLoading] = useState(true);
  const [videoProgress, setVideoProgress] = useState(0);

  // Check video readiness
  const checkVideoReady = async () => {
    try {
      const res = await fetch(`${API_URL}/api/status`);
      const data = await res.json();
      
      if (data.video_cache) {
        const { is_downloading, download_progress, error } = data.video_cache;
        
        // Show loading while downloading
        setVideoLoading(is_downloading);
        setVideoProgress(download_progress || 0);
        
        // If there's an error, stop loading and show error
        if (error) {
          console.error('[Home] Video download error:', error);
          setVideoLoading(false);
          setError(`Video download failed: ${error}`);
          return;
        }
        
        // If still downloading, check again in 1 second
        if (is_downloading) {
          setTimeout(checkVideoReady, 1000);
        }
      } else {
        setVideoLoading(false);
      }
    } catch (e) {
      console.error('Failed to check video status:', e);
      setVideoLoading(false);
    }
  };

  const reload = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API_URL}/api/cameras`, { method: 'GET' });
      // If backend returns 404, show maintenance page
      if (res.status === 404) {
        setMaintenance(true);
        setLoading(false);
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      // Clear any previous maintenance flag on success
      setMaintenance(false);
      const data = await res.json();
      const cameraList = Array.isArray(data) ? data : [];
      setCameras(cameraList);
      
      // Pre-cache the first camera's video source
      if (cameraList.length > 0) {
        const firstCamera = cameraList[0];
        const link = firstCamera.link || firstCamera.cameraLink;
        if (link) {
          console.log('[Home] Pre-caching first camera:', link);
          try {
            await axios.post(`${API_URL}/api/set_video_source`, { source: link });
            console.log('[Home] Video source set, background download started');
            // Start polling for download status
            checkVideoReady();
          } catch (e) {
            console.error('[Home] Failed to set video source:', e);
            setVideoLoading(false);
          }
        } else {
          setVideoLoading(false);
        }
      } else {
        setVideoLoading(false);
      }
    } catch (e) {
      // When the browser blocks the response due to CORS (or other network failures),
      // fetch() will throw a TypeError ('Failed to fetch') and the response/status
      // won't be available. In that case show the maintenance page as a UX fallback.
      const msg = (e && e.message) ? e.message : '';
      const isNetworkOrCors = e instanceof TypeError || /failed to fetch|networkerror|network error/i.test(msg);

      if (isNetworkOrCors) {
        // Show maintenance UI when the request was blocked by CORS or network failure
        setMaintenance(true);
      } else {
        setError(`Failed to load cameras: ${msg}`);
      }
      setVideoLoading(false);
    } finally {
      setLoading(false);
    }
  };

  const handleAdd = (cam) => {
    // Optimistic add then reload from server to reflect real state
    setCameras((prev) => [cam, ...prev]);
    setTimeout(reload, 500);
  };

  const handleDeleteClick = (camera) => {
    setCameraToDelete(camera);
    setDeleteModalOpen(true);
  };

  const handleDeleteConfirmed = (cameraId) => {
    // Remove from local state immediately
    setCameras((prev) => prev.filter(c => (c.cameraId || c.id) !== cameraId));
    // Reload to sync with server
    setTimeout(reload, 500);
  };

  useEffect(() => {
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Function to get loading message based on progress
  const getLoadingMessage = (progress) => {
    if (progress < 20) {
      return "Connecting to live stream...";
    } else if (progress < 40) {
      return "Establishing secure connection...";
    } else if (progress < 60) {
      return "Setting up environment...";
    } else if (progress < 80) {
      return "Initializing video stream...";
    } else {
      return "Almost ready...";
    }
  };

  return (
    <>
      <Header />
      {maintenance ? (
        <div className="home">
          <MaintenancePage />
        </div>
      ) : (
        <div className="home">
          {/* Video Loading Overlay */}
          {videoLoading && (
            <div className="loading-overlay">
              <div className="loading-content">
                <div className="spinner"></div>
                <h2>Preparing Live Stream</h2>
                <p>{getLoadingMessage(videoProgress)}</p>
                <div className="progress-bar">
                  <div className="progress-fill" style={{width: `${videoProgress}%`}}></div>
                </div>
                <p style={{fontSize: '14px', opacity: 0.7, marginTop: '8px'}}>{videoProgress}%</p>
              </div>
            </div>
          )}

          <div className="cards-container">
            <section className="cards-grid">
              {loading && (
                <div className="empty-state">
                  <div className="spinner"></div>
                  <p>Loading cameras…</p>
                </div>
              )}
              {!loading && error && (
                <div className="empty-state"><p>{error}</p></div>
              )}
              {!loading && !error && cameras.map((cam) => (
                <CameraCard 
                  key={cam.cameraId || cam.id} 
                  camera={cam} 
                  onDelete={handleDeleteClick}
                />
              ))}
              {/* Always show Add Camera card at the end */}
              {!loading && !error && (
                <div className="add-camera-card" onClick={() => setModalOpen(true)}>
                  <div className="plus-icon">+</div>
                  <div className="add-text">ADD CAMERA</div>
                </div>
              )}
            </section>
          </div>

          <AddCameraModal 
            open={modalOpen} 
            onClose={() => setModalOpen(false)} 
            onAdd={handleAdd} 
            onSaved={reload} 
          />
          
          <DeleteCameraModal
            open={deleteModalOpen}
            onClose={() => setDeleteModalOpen(false)}
            camera={cameraToDelete}
            onDeleted={handleDeleteConfirmed}
          />
        </div>
      )}
    </>
  );
}
