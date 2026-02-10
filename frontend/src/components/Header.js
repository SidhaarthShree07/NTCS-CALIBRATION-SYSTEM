import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './Header.css';

export default function Header() {
  const navigate = useNavigate();
  const location = useLocation();
  
  const isHome = location.pathname === '/';
  const isCalibration = location.pathname === '/calibration';
  const isViolations = location.pathname === '/violations';
  const isLocations = location.pathname === '/locations';
  
  return (
    <header className="ntcs-header">
      <div className="header-left">
        <div className="logo">
          <span className="logo-icon">
            <img src="/ntcs.svg" alt="NTCS" />
          </span>
          <span className="logo-text">NTCS</span>
        </div>
      </div>
      
      <div className="header-right">
        <button 
          className={`nav-btn ${isHome ? 'active' : ''}`}
          onClick={() => navigate('/')}
        >
          Home
        </button>
        <button 
          className={`nav-btn ${isCalibration ? 'active' : ''}`}
          onClick={() => navigate('/calibration')}
        >
          Cameras
        </button>
        <button 
          className={`nav-btn ${isViolations ? 'active' : ''}`}
          onClick={() => navigate('/violations')}
        >
          Violations
        </button>
        <button 
          className={`nav-btn ${isLocations ? 'active' : ''}`}
          onClick={() => navigate('/locations')}
        >
          Locations
        </button>
      </div>
    </header>
  );
}
