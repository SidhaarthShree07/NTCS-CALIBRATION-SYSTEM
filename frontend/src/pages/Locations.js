import React, { useEffect, useState } from 'react';
import Header from '../components/Header';
import API_URL from '../config';
import '../components/Modal.css';
import './Locations.css';

export default function Locations() {
  const [locations, setLocations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [modalOpen, setModalOpen] = useState(false);
  const [editing, setEditing] = useState(null);
  const [name, setName] = useState('');
  const [speedLimit, setSpeedLimit] = useState('');
  const [saving, setSaving] = useState(false);

  const resetForm = () => {
    setName('');
    setSpeedLimit('');
    setEditing(null);
  };

  const loadLocations = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API_URL}/api/locations`);
      const data = await res.json().catch(() => ({}));
      const list = Array.isArray(data.locations) ? data.locations : [];
      setLocations(list);
    } catch (err) {
      setError(`Failed to load locations: ${err.message || err}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadLocations();
  }, []);

  const openAdd = () => {
    resetForm();
    setModalOpen(true);
  };

  const openEdit = (loc) => {
    setEditing(loc);
    setName(loc.name || '');
    setSpeedLimit(String(loc.speedLimit ?? ''));
    setModalOpen(true);
  };

  const handleSave = async (e) => {
    e.preventDefault();
    setError('');

    const trimmed = name.trim();
    const parsedSpeed = parseInt(speedLimit, 10);

    if (!trimmed) {
      setError('Location name is required.');
      return;
    }
    if (!Number.isFinite(parsedSpeed) || parsedSpeed <= 0) {
      setError('Speed limit must be a positive number.');
      return;
    }

    setSaving(true);
    try {
      const payload = { name: trimmed, speedLimit: parsedSpeed };
      const url = editing
        ? `${API_URL}/api/locations/${editing.locationId}`
        : `${API_URL}/api/locations`;
      const method = editing ? 'PUT' : 'POST';

      const res = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data?.message || `HTTP ${res.status}`);
      }

      setModalOpen(false);
      resetForm();
      loadLocations();
    } catch (err) {
      setError(`Failed to save location: ${err.message || err}`);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (loc) => {
    if (!window.confirm(`Delete location "${loc.name}"?`)) return;
    setError('');
    try {
      const res = await fetch(`${API_URL}/api/locations/${loc.locationId}`, {
        method: 'DELETE',
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data?.message || `HTTP ${res.status}`);
      }
      loadLocations();
    } catch (err) {
      setError(`Failed to delete location: ${err.message || err}`);
    }
  };

  return (
    <>
      <Header />
      <div className="locations-page">
        <div className="locations-header">
          <div>
            <h1>Locations</h1>
            <p>Manage locations and their speed limits.</p>
          </div>
          <button className="btn primary" onClick={openAdd}>Add Location</button>
        </div>

        {error && <div className="locations-error">{error}</div>}

        <div className="locations-card">
          {loading ? (
            <div className="locations-empty">Loading locations...</div>
          ) : locations.length === 0 ? (
            <div className="locations-empty">No locations yet. Add one to get started.</div>
          ) : (
            <table className="locations-table">
              <thead>
                <tr>
                  <th>Location</th>
                  <th>Speed Limit</th>
                  <th className="actions-col">Actions</th>
                </tr>
              </thead>
              <tbody>
                {locations.map((loc) => (
                  <tr key={loc.locationId}>
                    <td>{loc.name}</td>
                    <td>{loc.speedLimit} km/h</td>
                    <td className="actions-col">
                      <button className="btn" onClick={() => openEdit(loc)}>Edit</button>
                      <button className="btn danger" onClick={() => handleDelete(loc)}>Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {modalOpen && (
        <div className="modal-backdrop" onClick={() => setModalOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>{editing ? 'Edit Location' : 'Add Location'}</h2>
              <button className="icon-btn" onClick={() => setModalOpen(false)} aria-label="Close">×</button>
            </div>
            <form className="modal-body" onSubmit={handleSave}>
              <label>
                Location Name
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g., Football Chowk"
                  disabled={saving}
                  required
                />
              </label>

              <label>
                Speed Limit (km/h)
                <input
                  type="number"
                  min="1"
                  value={speedLimit}
                  onChange={(e) => setSpeedLimit(e.target.value)}
                  placeholder="e.g., 60"
                  disabled={saving}
                  required
                />
              </label>

              {error && <div className="form-error" role="alert">{error}</div>}

              <div className="modal-footer">
                <button type="button" className="btn" onClick={() => setModalOpen(false)} disabled={saving}>Cancel</button>
                <button type="submit" className="btn primary" disabled={saving}>
                  {saving ? 'Saving…' : 'Save'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
