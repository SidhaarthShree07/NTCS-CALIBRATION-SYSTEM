import React from 'react';
import { motion } from 'framer-motion';
import './Button.css';

export const Button = ({
  children,
  variant = 'primary', // primary, secondary, ghost
  size = 'md', // sm, md, lg
  icon,
  iconPosition = 'left',
  loading = false,
  disabled = false,
  className = '',
  onClick,
  ...props
}) => {
  return (
    <motion.button
      className={`ntcs-button ntcs-button-${variant} ntcs-button-${size} ${className}`}
      onClick={onClick}
      disabled={disabled || loading}
      whileHover={{ scale: disabled ? 1 : 1.02, y: disabled ? 0 : -2 }}
      whileTap={{ scale: disabled ? 1 : 0.98 }}
      transition={{ duration: 0.15 }}
      {...props}
    >
      {loading && (
        <div className="spinner"></div>
      )}
      {!loading && icon && iconPosition === 'left' && (
        <span className="button-icon">{icon}</span>
      )}
      <span>{children}</span>
      {!loading && icon && iconPosition === 'right' && (
        <span className="button-icon">{icon}</span>
      )}
    </motion.button>
  );
};

export default Button;
