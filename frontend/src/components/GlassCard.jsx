import React from 'react';
import { motion } from 'framer-motion';
import './GlassCard.css';

export const GlassCard = ({ 
  children, 
  className = '', 
  hover = true,
  glow = false,
  onClick,
  ...props 
}) => {
  return (
    <motion.div
      className={`glass-card ${glow ? 'glow-effect' : ''} ${className}`}
      onClick={onClick}
      whileHover={hover ? { scale: 1.02, y: -4 } : {}}
      transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
      {...props}
    >
      {children}
    </motion.div>
  );
};

export default GlassCard;
