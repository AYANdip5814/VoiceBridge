import React from 'react';
import PropTypes from 'prop-types';

const Logo = ({ size = 'medium', className = '', showText = true }) => {
  // Size mapping
  const sizeMap = {
    small: { width: 40, height: 40, fontSize: 'text-sm' },
    medium: { width: 60, height: 60, fontSize: 'text-lg' },
    large: { width: 80, height: 80, fontSize: 'text-xl' }
  };

  const { width, height, fontSize } = sizeMap[size] || sizeMap.medium;

  return (
    <div className={`flex items-center ${className}`}>
      <div className="relative" style={{ width, height }}>
        {/* Bridge symbol */}
        <svg 
          viewBox="0 0 100 100" 
          className="w-full h-full"
          aria-hidden="true"
        >
          {/* Bridge arch */}
          <path 
            d="M10,80 Q50,20 90,80" 
            fill="none" 
            stroke="#8B5CF6" 
            strokeWidth="6" 
            strokeLinecap="round"
          />
          
          {/* Bridge supports */}
          <path 
            d="M10,80 L10,90 M90,80 L90,90" 
            fill="none" 
            stroke="#8B5CF6" 
            strokeWidth="6" 
            strokeLinecap="round"
          />
          
          {/* Hand gesture symbol */}
          <circle 
            cx="50" 
            cy="50" 
            r="25" 
            fill="#6D28D9" 
            opacity="0.8"
          />
          
          {/* Hand fingers */}
          <path 
            d="M50,35 L50,25 M40,40 L30,35 M60,40 L70,35" 
            fill="none" 
            stroke="white" 
            strokeWidth="4" 
            strokeLinecap="round"
          />
          
          {/* Sound waves */}
          <path 
            d="M75,50 Q80,45 85,50 M80,50 Q85,45 90,50" 
            fill="none" 
            stroke="white" 
            strokeWidth="2" 
            strokeLinecap="round"
          />
        </svg>
      </div>
      
      {showText && (
        <div className={`ml-2 font-bold ${fontSize} text-white`}>
          <span className="text-purple-300">Voice</span>
          <span className="text-indigo-300">Bridge</span>
        </div>
      )}
    </div>
  );
};

Logo.propTypes = {
  size: PropTypes.oneOf(['small', 'medium', 'large']),
  className: PropTypes.string,
  showText: PropTypes.bool
};

export default Logo; 