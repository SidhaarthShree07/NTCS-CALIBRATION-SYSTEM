import React, { useEffect, useState, useCallback } from 'react';
import Header from '../components/Header';
import API_URL from '../config';
import './Violations.css';

export default function Violations() {
  const [violations, setViolations] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [searchPlate, setSearchPlate] = useState('');
  const [filterCamera, setFilterCamera] = useState('');
  const [cameras, setCameras] = useState([]);
  const [selectedViolation, setSelectedViolation] = useState(null);
  const [modalOpen, setModalOpen] = useState(false);

  const perPage = 12;

  // Fetch cameras for filter dropdown
  useEffect(() => {
    fetch(`${API_URL}/api/cameras`)
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) setCameras(data);
      })
      .catch(() => {});
  }, []);

  // Fetch violation stats
  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/violations/stats`);
      const data = await res.json();
      setStats(data);
    } catch (e) {
      console.error('Failed to fetch stats:', e);
    }
  }, []);

  // Fetch violations
  const fetchViolations = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ page, per_page: perPage });
      if (searchPlate) params.append('plate', searchPlate);
      if (filterCamera) params.append('camera_id', filterCamera);

      const res = await fetch(`${API_URL}/api/violations?${params}`);
      const data = await res.json();

      setViolations(data.violations || []);
      setTotalPages(data.pages || 1);
      setTotal(data.total || 0);
    } catch (e) {
      console.error('Failed to fetch violations:', e);
      setViolations([]);
    } finally {
      setLoading(false);
    }
  }, [page, searchPlate, filterCamera]);

  useEffect(() => {
    fetchViolations();
    fetchStats();
  }, [fetchViolations, fetchStats]);

  // Convert file:// and Azure blob URLs to proxied API URLs
  const resolveUrl = (url) => {
    if (!url) return null;
    if (url.startsWith('file://')) {
      // Extract filename from file:// path
      const parts = url.replace('file://', '').replace(/\\/g, '/').split('/');
      const filename = parts[parts.length - 1];
      return `${API_URL}/api/proof/${filename}`;
    }
    // Proxy Azure blob URLs through backend to avoid CORS/private access issues
    if (url.includes('blob.core.windows.net')) {
      return `${API_URL}/api/blob-proxy?url=${encodeURIComponent(url)}`;
    }
    return url;
  };

  const openDetail = (v) => {
    setSelectedViolation(v);
    setModalOpen(true);
  };

  const closeModal = () => {
    setModalOpen(false);
    setSelectedViolation(null);
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A';
    try {
      const d = new Date(dateStr);
      return d.toLocaleString('en-IN', {
        day: '2-digit', month: 'short', year: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: true
      });
    } catch {
      return dateStr;
    }
  };

  const getSpeedColor = (measured, limit) => {
    if (!measured || !limit) return '';
    const ratio = measured / limit;
    if (ratio >= 1.5) return 'speed-critical';
    if (ratio >= 1.25) return 'speed-high';
    return 'speed-over';
  };

  return (
    <>
      <Header />
      <div className="violations-page">
        {/* Stats Cards */}
        {stats && (
          <div className="stats-row">
            <div className="stat-card">
              <div className="stat-value">{stats.totalViolations}</div>
              <div className="stat-label">Total Violations</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{stats.todayViolations}</div>
              <div className="stat-label">Today</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{stats.avgSpeed} <span className="stat-unit">km/h</span></div>
              <div className="stat-label">Avg Speed</div>
            </div>
            <div className="stat-card">
              <div className="stat-value speed-critical">{stats.maxSpeed} <span className="stat-unit">km/h</span></div>
              <div className="stat-label">Max Speed</div>
            </div>
          </div>
        )}

        {/* Filters */}
        <div className="filters-row">
          <div className="filter-group">
            <input
              type="text"
              className="filter-input"
              placeholder="Search plate number..."
              value={searchPlate}
              onChange={(e) => { setSearchPlate(e.target.value); setPage(1); }}
            />
          </div>
          <div className="filter-group">
            <select
              className="filter-select"
              value={filterCamera}
              onChange={(e) => { setFilterCamera(e.target.value); setPage(1); }}
            >
              <option value="">All Cameras</option>
              {cameras.map(c => (
                <option key={c.cameraId} value={c.cameraId}>{c.cameraId} - {c.location}</option>
              ))}
            </select>
          </div>
          <div className="filter-count">
            {total} violation{total !== 1 ? 's' : ''} found
          </div>
        </div>

        {/* Violations Grid */}
        {loading ? (
          <div className="violations-loading">
            <div className="spinner"></div>
            <p>Loading violations...</p>
          </div>
        ) : violations.length === 0 ? (
          <div className="violations-empty">
            <div className="empty-icon">🚗</div>
            <h3>No Violations Found</h3>
            <p>No speed violations have been recorded yet.</p>
          </div>
        ) : (
          <div className="violations-grid">
            {violations.map((v) => (
              <div key={v.eventId} className="violation-card" onClick={() => openDetail(v)}>
                <div className="violation-image">
                  {resolveUrl(v.evidence?.imageOriginalUrl) ? (
                    <img
                      src={resolveUrl(v.evidence.imageOriginalUrl)}
                      alt="Violation"
                      onError={(e) => { e.target.style.display = 'none'; }}
                    />
                  ) : (
                    <div className="no-image">No Image</div>
                  )}
                  <div className={`speed-badge ${getSpeedColor(v.violation?.measured, v.violation?.limit)}`}>
                    {v.violation?.measured?.toFixed(0)} km/h
                  </div>
                </div>
                <div className="violation-info">
                  <div className="violation-plate">
                    {v.vehicle?.plate?.text || 'Plate N/A'}
                  </div>
                  <div className="violation-meta">
                    <span className="meta-item">
                      <span className="meta-icon">📷</span>
                      {v.cameraId || 'Unknown'}
                    </span>
                    <span className="meta-item">
                      <span className="meta-icon">🕐</span>
                      {formatDate(v.capturedAt)}
                    </span>
                  </div>
                  <div className="violation-details-row">
                    <span className="detail-chip vehicle-type">
                      {v.vehicle?.vehicleClass || 'VEHICLE'}
                    </span>
                    <span className="detail-chip speed-limit">
                      Limit: {v.violation?.limit?.toFixed(0)} km/h
                    </span>
                    {v.vehicle?.plate?.confidence > 0 && (
                      <span className="detail-chip confidence">
                        {(v.vehicle.plate.confidence * 100).toFixed(0)}% OCR
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="pagination">
            <button
              className="page-btn"
              disabled={page <= 1}
              onClick={() => setPage(p => Math.max(1, p - 1))}
            >
              ← Prev
            </button>
            <span className="page-info">Page {page} of {totalPages}</span>
            <button
              className="page-btn"
              disabled={page >= totalPages}
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            >
              Next →
            </button>
          </div>
        )}

        {/* Detail Modal */}
        {modalOpen && selectedViolation && (
          <div className="violation-modal-overlay" onClick={closeModal}>
            <div className="violation-modal" onClick={(e) => e.stopPropagation()}>
              <button className="modal-close" onClick={closeModal}>×</button>
              <div className="modal-content">
                <div className="modal-left">
                  {/* Original Screenshot */}
                  {resolveUrl(selectedViolation.evidence?.imageOriginalUrl) && (
                    <div className="modal-image-section">
                      <h4>Original Capture</h4>
                      <img
                        src={resolveUrl(selectedViolation.evidence.imageOriginalUrl)}
                        alt="Original"
                        className="modal-image"
                      />
                    </div>
                  )}
                  {/* Enhanced Plate */}
                  {resolveUrl(selectedViolation.evidence?.imageEnhancedUrl) && (
                    <div className="modal-image-section">
                      <h4>Enhanced Plate</h4>
                      <img
                        src={resolveUrl(selectedViolation.evidence.imageEnhancedUrl)}
                        alt="Enhanced Plate"
                        className="modal-image plate-image"
                      />
                    </div>
                  )}
                  {/* Video Clip */}
                  {resolveUrl(selectedViolation.evidence?.videoClipUrl) && (
                    <div className="modal-image-section">
                      <h4>Video Evidence</h4>
                      <video
                        src={resolveUrl(selectedViolation.evidence.videoClipUrl)}
                        controls
                        className="modal-video"
                        autoPlay
                        loop
                        muted
                      />
                    </div>
                  )}
                </div>
                <div className="modal-right">
                  <h2>Violation Details</h2>

                  <div className="detail-section">
                    <h4>Speed Violation</h4>
                    <div className="detail-grid">
                      <div className="detail-item">
                        <span className="detail-label">Measured Speed</span>
                        <span className={`detail-value ${getSpeedColor(selectedViolation.violation?.measured, selectedViolation.violation?.limit)}`}>
                          {selectedViolation.violation?.measured?.toFixed(1)} km/h
                        </span>
                      </div>
                      <div className="detail-item">
                        <span className="detail-label">Speed Limit</span>
                        <span className="detail-value">{selectedViolation.violation?.limit?.toFixed(0)} km/h</span>
                      </div>
                      <div className="detail-item">
                        <span className="detail-label">Over Limit By</span>
                        <span className="detail-value speed-critical">
                          +{((selectedViolation.violation?.measured || 0) - (selectedViolation.violation?.limit || 0)).toFixed(1)} km/h
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="detail-section">
                    <h4>Vehicle Information</h4>
                    <div className="detail-grid">
                      <div className="detail-item">
                        <span className="detail-label">License Plate</span>
                        <span className="detail-value plate-value">
                          {selectedViolation.vehicle?.plate?.text || 'N/A'}
                        </span>
                      </div>
                      <div className="detail-item">
                        <span className="detail-label">Vehicle Type</span>
                        <span className="detail-value">{selectedViolation.vehicle?.vehicleClass || 'Unknown'}</span>
                      </div>
                      <div className="detail-item">
                        <span className="detail-label">OCR Confidence</span>
                        <span className="detail-value">
                          {selectedViolation.vehicle?.plate?.confidence
                            ? `${(selectedViolation.vehicle.plate.confidence * 100).toFixed(1)}%`
                            : 'N/A'}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="detail-section">
                    <h4>Event Info</h4>
                    <div className="detail-grid">
                      <div className="detail-item">
                        <span className="detail-label">Event ID</span>
                        <span className="detail-value mono">{selectedViolation.eventId}</span>
                      </div>
                      <div className="detail-item">
                        <span className="detail-label">Camera</span>
                        <span className="detail-value">{selectedViolation.cameraId}</span>
                      </div>
                      <div className="detail-item">
                        <span className="detail-label">Captured At</span>
                        <span className="detail-value">{formatDate(selectedViolation.capturedAt)}</span>
                      </div>
                      <div className="detail-item">
                        <span className="detail-label">Violation Type</span>
                        <span className="detail-value">{selectedViolation.violation?.type || 'OVERSPEED'}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
