import React, { useRef, useEffect, useImperativeHandle, forwardRef } from 'react';

const VideoPlayer = forwardRef(({ videoUrl, onTimeUpdate }, ref) => {
    const videoRef = useRef(null);

    // Expose jumpTo method to parent components
    useImperativeHandle(ref, () => ({
        jumpTo: (seconds) => {
            if (videoRef.current) {
                videoRef.current.currentTime = seconds;
                videoRef.current.play().catch(() => { });
            }
        },
        getCurrentTime: () => {
            return videoRef.current ? videoRef.current.currentTime : 0;
        },
    }));

    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        const handleTimeUpdate = () => {
            if (onTimeUpdate) {
                onTimeUpdate(video.currentTime);
            }
        };

        video.addEventListener('timeupdate', handleTimeUpdate);
        return () => video.removeEventListener('timeupdate', handleTimeUpdate);
    }, [onTimeUpdate]);

    if (!videoUrl) {
        return (
            <div className="player-container" style={{
                height: 360,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--text-muted)'
            }}>
                <div className="empty-state">
                    <div className="empty-state-icon">🎥</div>
                    <p>No video loaded</p>
                </div>
            </div>
        );
    }

    return (
        <div className="player-container">
            <video
                ref={videoRef}
                src={videoUrl}
                controls
                style={{ width: '100%', display: 'block', borderRadius: 'var(--radius-lg)' }}
            />
        </div>
    );
});

VideoPlayer.displayName = 'VideoPlayer';

export default VideoPlayer;
