import React, { useEffect, useState } from 'react';
import './Modal.css';
import API_URL from '../config';

export default function AddCameraModal({ open, onClose, onAdd, onSaved }) {
  const [cameraId, setCameraId] = useState('');
  const [cameraLink, setCameraLink] = useState('');
  const [location, setLocation] = useState('');
  const [locations, setLocations] = useState([]);
  const [locationSpeed, setLocationSpeed] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  const reset = () => {
    setCameraId('');
    setCameraLink('');
    setLocation('');
    setLocationSpeed(null);
  };

  useEffect(() => {
    if (!open) return;
    let isActive = true;
    const loadLocations = async () => {
      try {
        const res = await fetch(`${API_URL}/api/locations`);
        const data = await res.json().catch(() => ({}));
        const list = Array.isArray(data.locations) ? data.locations : [];
        if (!isActive) return;
        setLocations(list);
        if (list.length > 0) {
          setLocation(list[0].name);
          setLocationSpeed(list[0].speedLimit);
          setError('');
        } else {
          setLocation('');
          setLocationSpeed(null);
          setError('No locations found. Add a location first.');
        }
      } catch (err) {
        if (!isActive) return;
        setLocations([]);
        setLocation('');
        setLocationSpeed(null);
        setError('Failed to load locations. Add a location first.');
      }
    };
    loadLocations();
    return () => {
      isActive = false;
    };
  }, [open]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (!cameraId.trim() || !cameraLink.trim() || !location) return;
    const payload = { cameraId: cameraId.trim(), cameraLink: cameraLink.trim(), location };
    try {
      setSubmitting(true);
      // Use local proxy to avoid CORS
      const res = await fetch(`${API_URL}/api/calibhome`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data?.message || `HTTP ${res.status}`);
      }
  // Optimistically add to UI
      onAdd(payload);
  // Trigger a reload so UI reflects server-side source of truth
  if (typeof onSaved === 'function') onSaved();
      reset();
      onClose();
    } catch (err) {
      setError(`Failed to add camera: ${err.message || err}`);
    } finally {
      setSubmitting(false);
    }
  };

  if (!open) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Add Camera</h3>
          <button className="icon-btn" onClick={onClose} aria-label="Close">×</button>
        </div>
        <form className="modal-body" onSubmit={handleSubmit}>
          <label>
            Camera ID
            <input
              type="text"
              value={cameraId}
              onChange={(e) => setCameraId(e.target.value)}
              placeholder="e.g., CAM-001"
              required
              disabled={submitting}
            />
          </label>

          <label>
            Camera Link (URL)
            <input
              type="url"
              value={cameraLink}
              onChange={(e) => setCameraLink(e.target.value)}
              placeholder="https://..."
              required
              disabled={submitting}
            />
          </label>

          <label>
            Camera Location
            <select
              value={location}
              onChange={(e) => {
                const next = e.target.value;
                setLocation(next);
                const match = locations.find((l) => l.name === next);
                setLocationSpeed(match ? match.speedLimit : null);
              }}
              disabled={submitting || locations.length === 0}
            >
              {locations.length === 0 && (
                <option value="">No locations available</option>
              )}
              {locations.map((opt) => (
                <option key={opt.locationId} value={opt.name}>
                  {opt.name}
                </option>
              ))}
            </select>
          </label>

          <label>
            Speed Limit (from location)
            <input
              type="text"
              value={locationSpeed !== null ? `${locationSpeed} km/h` : 'Not set'}
              disabled
              readOnly
            />
          </label>

          {error && <div className="form-error" role="alert">{error}</div>}

          <div className="modal-footer">
            <button type="button" className="btn" onClick={onClose} disabled={submitting}>Cancel</button>
            <button
              type="submit"
              className="btn primary"
              disabled={submitting || !location}
            >
              {submitting ? 'Adding…' : 'Add Camera'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
