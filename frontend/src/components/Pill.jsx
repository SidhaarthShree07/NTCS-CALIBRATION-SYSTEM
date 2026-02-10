import React from 'react';
import { motion } from 'framer-motion';
import './Pill.css';

export const Pill = ({ children, icon, className = '' }) => {
  return (
    <motion.div
      className={`ntcs-pill ${className}`}
      whileHover={{ scale: 1.05 }}
      transition={{ duration: 0.2 }}
    >
      {icon && <span className="pill-icon">{icon}</span>}
      <span>{children}</span>
    </motion.div>
  );
};

export default Pill;
