import React from 'react';
import PropTypes from 'prop-types';

const LanguageSelector = ({ selectedLanguage, onLanguageChange, languages }) => {
  const handleChange = (event) => {
    try {
      const newLanguage = event.target.value;
      if (languages.some(lang => lang.code === newLanguage)) {
        onLanguageChange(newLanguage);
      } else {
        console.error('Invalid language selected:', newLanguage);
      }
    } catch (error) {
      console.error('Error changing language:', error);
    }
  };

  return (
    <div className="flex items-center space-x-2">
      <label 
        htmlFor="language-select" 
        className="text-white font-medium"
        aria-label="Select sign language"
      >
        Sign Language:
      </label>
      <select
        id="language-select"
        value={selectedLanguage}
        onChange={handleChange}
        className="bg-indigo-700 text-white border border-purple-500 rounded-md px-3 py-1.5 
                 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent
                 hover:bg-indigo-600 transition-colors"
        aria-label="Sign language selection"
      >
        {languages.map((lang) => (
          <option 
            key={lang.code} 
            value={lang.code}
            aria-label={lang.name}
          >
            {lang.name}
          </option>
        ))}
      </select>
    </div>
  );
};

LanguageSelector.propTypes = {
  selectedLanguage: PropTypes.string.isRequired,
  onLanguageChange: PropTypes.func.isRequired,
  languages: PropTypes.arrayOf(
    PropTypes.shape({
      code: PropTypes.string.isRequired,
      name: PropTypes.string.isRequired
    })
  ).isRequired
};

export default LanguageSelector; 