import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import UploadZone from '../components/UploadZone';
import { listVideos, getVideoStatus } from '../services/api';

export default function HomePage() {
    const [videos, setVideos] = useState([]);
    const [processingIds, setProcessingIds] = useState([]);
    const navigate = useNavigate();

    // Load videos on mount
    useEffect(() => {
        loadVideos();
    }, []);

    // Poll processing videos
    useEffect(() => {
        if (processingIds.length === 0) return;

        const interval = setInterval(async () => {
            for (const id of processingIds) {
                try {
                    const status = await getVideoStatus(id);
                    setVideos((prev) =>
                        prev.map((v) => (v.video_id === id ? status : v))
                    );
                    if (status.status === 'ready' || status.status === 'error') {
                        setProcessingIds((prev) => prev.filter((pid) => pid !== id));
                    }
                } catch (e) {
                    console.error('Status poll error:', e);
                }
            }
        }, 2000);

        return () => clearInterval(interval);
    }, [processingIds]);

    const loadVideos = async () => {
        try {
            const data = await listVideos();
            setVideos(data.videos || []);
        } catch (e) {
            console.log('Backend not available yet');
        }
    };

    const handleUploadComplete = (video) => {
        setVideos((prev) => [video, ...prev]);
        setProcessingIds((prev) => [...prev, video.video_id]);
    };

    const handleVideoClick = (video) => {
        if (video.status === 'ready') {
            navigate(`/video/${video.video_id}`);
        }
    };

    const getStatusInfo = (status) => {
        const map = {
            uploading: { cls: 'status-processing', label: '⬆️ Uploading' },
            extracting: { cls: 'status-processing', label: '🎞️ Extracting' },
            transcribing: { cls: 'status-processing', label: '🎤 Transcribing' },
            embedding: { cls: 'status-processing', label: '🧮 Embedding' },
            indexing: { cls: 'status-processing', label: '📦 Indexing' },
            ready: { cls: 'status-ready', label: '✅ Ready' },
            error: { cls: 'status-error', label: '❌ Error' },
        };
        return map[status] || map.uploading;
    };

    return (
        <div className="home-page">
            {/* Hero */}
            <div className="home-hero">
                <h1>🧠 Video Brain</h1>
                <p>
                    Upload any video and unlock AI-powered semantic search, visual reasoning,
                    and natural language chat — all running on your GPU.
                </p>
            </div>

            {/* Upload */}
            <UploadZone onUploadComplete={handleUploadComplete} />

            {/* Video Library */}
            {videos.length > 0 && (
                <>
                    <h2 className="section-title">Your Videos</h2>
                    <div className="video-grid">
                        {videos.map((video) => {
                            const statusInfo = getStatusInfo(video.status);
                            return (
                                <div
                                    key={video.video_id}
                                    className="video-card"
                                    onClick={() => handleVideoClick(video)}
                                    style={{ opacity: video.status === 'ready' ? 1 : 0.7 }}
                                >
                                    <div className="video-card-thumb">
                                        🎬
                                    </div>
                                    <div className="video-card-body">
                                        <div className="video-card-title">
                                            {video.filename || `Video ${video.video_id}`}
                                        </div>
                                        <div className="video-card-meta">
                                            <span className={`status-badge ${statusInfo.cls}`}>
                                                {statusInfo.label}
                                            </span>
                                            {video.duration && (
                                                <span>{Math.round(video.duration)}s</span>
                                            )}
                                            {video.frame_count && (
                                                <span>{video.frame_count} frames</span>
                                            )}
                                        </div>
                                        {video.status !== 'ready' && video.status !== 'error' && (
                                            <div style={{ marginTop: 8 }}>
                                                <div className="progress-bar-bg">
                                                    <div
                                                        className="progress-bar-fill"
                                                        style={{ width: `${(video.progress || 0) * 100}%` }}
                                                    />
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </>
            )}
        </div>
    );
}
