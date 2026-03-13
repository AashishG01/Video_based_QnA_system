import React, { useState, useRef, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import VideoPlayer from '../components/VideoPlayer';
import ChatWindow from '../components/ChatWindow';
import SearchBar from '../components/SearchBar';
import { getVideoStatus, getTranscript } from '../services/api';

function formatTimestamp(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

export default function VideoPage() {
    const { videoId } = useParams();
    const navigate = useNavigate();
    const playerRef = useRef(null);
    const [videoInfo, setVideoInfo] = useState(null);
    const [transcript, setTranscript] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadVideoInfo();
    }, [videoId]);

    const loadVideoInfo = async () => {
        try {
            const info = await getVideoStatus(videoId);
            setVideoInfo(info);

            if (info.status === 'ready') {
                try {
                    const transcriptData = await getTranscript(videoId);
                    setTranscript(transcriptData.segments || []);
                } catch (e) {
                    console.log('Transcript not available');
                }
            }
        } catch (e) {
            console.error('Failed to load video:', e);
            navigate('/');
        } finally {
            setLoading(false);
        }
    };

    const jumpToTimestamp = (seconds) => {
        if (playerRef.current) {
            playerRef.current.jumpTo(seconds);
        }
    };

    if (loading) {
        return (
            <div className="empty-state" style={{ padding: 80 }}>
                <span className="spinner" style={{ width: 32, height: 32 }} />
                <p style={{ marginTop: 16 }}>Loading video...</p>
            </div>
        );
    }

    // Build the video URL from backend static files
    const videoUrl = videoInfo
        ? `/data/uploads/${videoId}/${videoInfo.filename}`
        : null;

    return (
        <div className="video-page">
            {/* LEFT: Video + Search + Transcript */}
            <div className="video-main">
                {/* Back button */}
                <div style={{ marginBottom: 16 }}>
                    <button
                        onClick={() => navigate('/')}
                        style={{
                            background: 'var(--bg-tertiary)',
                            color: 'var(--text-secondary)',
                            padding: '8px 16px',
                            borderRadius: 'var(--radius-sm)',
                            fontSize: 13,
                            border: '1px solid var(--border)',
                        }}
                    >
                        ← Back to Library
                    </button>
                    <span style={{
                        marginLeft: 16,
                        fontSize: 18,
                        fontWeight: 600,
                    }}>
                        {videoInfo?.filename || 'Video'}
                    </span>
                </div>

                {/* Video Player */}
                <VideoPlayer ref={playerRef} videoUrl={videoUrl} />

                {/* Search */}
                <SearchBar
                    videoId={videoId}
                    onResultClick={jumpToTimestamp}
                />

                {/* Transcript */}
                {transcript.length > 0 && (
                    <div className="transcript-panel">
                        <div className="transcript-header">
                            <span className="transcript-title">📝 Transcript</span>
                            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                                {transcript.length} segments
                            </span>
                        </div>
                        <div className="transcript-list">
                            {transcript.map((seg, i) => (
                                <div
                                    key={i}
                                    className="transcript-line"
                                    onClick={() => jumpToTimestamp(seg.start_time)}
                                >
                                    <span className="transcript-time">
                                        {formatTimestamp(seg.start_time)}
                                    </span>
                                    <span className="transcript-text">{seg.text}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            {/* RIGHT: Chat Sidebar */}
            <div className="video-sidebar">
                <ChatWindow
                    videoId={videoId}
                    onTimestampClick={jumpToTimestamp}
                />
            </div>
        </div>
    );
}
