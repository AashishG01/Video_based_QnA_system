import React, { useState } from 'react';
import { searchVideo } from '../services/api';

function formatTimestamp(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

export default function SearchBar({ videoId, onResultClick }) {
    const [query, setQuery] = useState('');
    const [searchType, setSearchType] = useState('hybrid');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [searched, setSearched] = useState(false);

    const handleSearch = async () => {
        if (!query.trim() || !videoId || loading) return;

        setLoading(true);
        setSearched(true);
        try {
            const data = await searchVideo(query.trim(), videoId, searchType);
            setResults(data.results || []);
        } catch (err) {
            console.error('Search error:', err);
            setResults([]);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') handleSearch();
    };

    const getSourceBadge = (type) => {
        const labels = {
            visual: { emoji: '🖼️', label: 'Visual', cls: 'source-visual' },
            text: { emoji: '🔊', label: 'Audio', cls: 'source-text' },
            hybrid: { emoji: '🔀', label: 'Hybrid', cls: 'source-hybrid' },
        };
        const info = labels[type] || labels.hybrid;
        return (
            <span className={`source-badge ${info.cls}`}>
                {info.emoji} {info.label}
            </span>
        );
    };

    return (
        <div className="search-section">
            {/* Search Type Tabs */}
            <div className="search-tabs">
                {['hybrid', 'visual', 'text'].map((type) => (
                    <button
                        key={type}
                        className={`search-tab ${searchType === type ? 'active' : ''}`}
                        onClick={() => setSearchType(type)}
                    >
                        {type === 'hybrid' ? '🔀 Hybrid' : type === 'visual' ? '🖼️ Visual' : '🔊 Text'}
                    </button>
                ))}
            </div>

            {/* Search Input */}
            <div className="search-bar">
                <input
                    id="search-input"
                    className="search-input"
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={videoId
                        ? 'Search video semantically... (e.g. "person writing on whiteboard")'
                        : 'Upload a video first...'
                    }
                    disabled={!videoId}
                />
                <button
                    id="search-btn"
                    className="search-btn"
                    onClick={handleSearch}
                    disabled={!query.trim() || !videoId || loading}
                >
                    {loading ? <span className="spinner" /> : '🔍 Search'}
                </button>
            </div>

            {/* Results */}
            {searched && (
                <div className="results-list">
                    {results.length === 0 && !loading ? (
                        <div className="empty-state" style={{ padding: 24 }}>
                            <p>No results found for "{query}"</p>
                        </div>
                    ) : (
                        results.map((result, i) => (
                            <div
                                key={i}
                                className="result-card"
                                onClick={() => onResultClick && onResultClick(result.timestamp)}
                            >
                                {result.frame_url && (
                                    <img
                                        className="result-thumb"
                                        src={result.frame_url}
                                        alt={`Frame at ${result.timestamp}s`}
                                        onError={(e) => { e.target.style.display = 'none'; }}
                                    />
                                )}
                                <div className="result-info">
                                    <span className="result-timestamp">
                                        ⏱ {formatTimestamp(result.timestamp)}
                                    </span>
                                    {result.transcript_text && (
                                        <div className="result-text">{result.transcript_text}</div>
                                    )}
                                    <div className="result-meta">
                                        {getSourceBadge(result.source_type)}
                                        <span>Score: {(result.score * 100).toFixed(1)}%</span>
                                    </div>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            )}
        </div>
    );
}
