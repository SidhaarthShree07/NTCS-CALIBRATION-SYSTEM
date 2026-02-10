import React from 'react';
import './App.css';
import { Routes, Route, Navigate } from 'react-router-dom';
import Home from './pages/Home';
import Calibration from './pages/Calibration';
import Violations from './pages/Violations';
import Locations from './pages/Locations';

// App component that defines routes
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/calibration" element={<Calibration />} />
      <Route path="/violations" element={<Violations />} />
      <Route path="/locations" element={<Locations />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
