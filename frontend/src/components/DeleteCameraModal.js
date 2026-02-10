import React, { useState } from 'react';
import axios from 'axios';
import './Modal.css'; // Reuse the same modal styles
import API_URL from '../config';

export default function DeleteCameraModal({ open, onClose, camera, onDeleted }) {
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState('');

  if (!open || !camera) return null;

  const handleDelete = async () => {
    setDeleting(true);
    setError('');

    try {
      const cameraId = camera.cameraId || camera.id;
      await axios.delete(`${API_URL}/api/calibhome/${cameraId}`);
      
      // Success - notify parent and close
      if (onDeleted) onDeleted(cameraId);
      onClose();
    } catch (err) {
      console.error('Delete failed:', err);
      setError(err.response?.data?.error || err.message || 'Failed to delete camera');
    } finally {
      setDeleting(false);
    }
  };

  const cameraId = camera.cameraId || camera.id;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 style={{ margin: 0 }}>Delete Camera</h2>
          <button className="icon-btn" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body">
          <p style={{ marginBottom: '10px', color: '#e8eaed' }}>
            Are you sure you want to delete camera <strong>{cameraId}</strong> at <strong>{camera.location}</strong>?
            <br />
            <span style={{ color: '#f28b82', fontSize: '0.9em' }}>This action cannot be undone.</span>
          </p>

          {error && <div className="error-msg" style={{ 
            padding: '10px', 
            background: '#4a2525', 
            border: '1px solid #8a4040', 
            borderRadius: '6px', 
            color: '#f28b82' 
          }}>{error}</div>}
        </div>
        <div className="modal-footer">
          <button 
            className="btn" 
            onClick={onClose} 
            disabled={deleting}
          >
            Cancel
          </button>
          <button 
            className="btn danger" 
            onClick={handleDelete} 
            disabled={deleting}
            style={{
              background: deleting ? '#4a2525' : '#c53030',
              borderColor: deleting ? '#8a4040' : '#f56565',
              color: '#fff'
            }}
          >
            {deleting ? 'Deleting...' : 'Delete Camera'}
          </button>
        </div>
      </div>
    </div>
  );
}
