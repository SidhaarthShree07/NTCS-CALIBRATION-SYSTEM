import React from 'react';
import './Maintenance.css';

export default function MaintenancePage() {
  return (
    <div className="maintenance-root">
      <div className="maintenance-card">
        <h1 className="maintenance-title">Service Temporarily Unavailable</h1>
        <p className="maintenance-sub">Our backend service is currently down because the Azure subscription credits have been exhausted.</p>
        <p className="maintenance-body">We're working to restore the service — it should be running again within the next week. Thank you for your patience.</p>
        <div className="maintenance-meta">
          <span className="meta-label">Status:</span>
          <span className="meta-value">Backend 404 — awaiting Azure top-up</span>
        </div>
      </div>
    </div>
  );
}
