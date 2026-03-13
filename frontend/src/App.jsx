import React from 'react';
import { Routes, Route, Link } from 'react-router-dom';
import HomePage from './pages/HomePage';
import VideoPage from './pages/VideoPage';

export default function App() {
    return (
        <>
            {/* Animated Background */}
            <div className="app-bg" />

            {/* Navbar */}
            <nav className="navbar">
                <Link to="/" className="navbar-brand">
                    <span className="navbar-logo">🧠</span>
                    <div>
                        <div className="navbar-title">Video Brain</div>
                        <div className="navbar-subtitle">Multi-Modal AI Video Intelligence</div>
                    </div>
                </Link>
            </nav>

            {/* Routes */}
            <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/video/:videoId" element={<VideoPage />} />
            </Routes>
        </>
    );
}
