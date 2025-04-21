import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import './Navigation.css';

const Navigation = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const token = localStorage.getItem('token');

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/auth');
  };

  if (!token) return null;

  return (
    <nav className="nav-container">
      <div className="container nav-content">
        <Link to="/" className="nav-logo">
          VoiceBridge
        </Link>
        <div className="nav-links">
          <Link
            to="/"
            className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
          >
            Home
          </Link>
          <Link
            to="/profile"
            className={`nav-link ${location.pathname === '/profile' ? 'active' : ''}`}
          >
            Profile
          </Link>
          <button onClick={handleLogout} className="nav-link logout-button">
            Logout
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navigation; 