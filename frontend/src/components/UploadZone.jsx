import React, { useState, useRef } from 'react';
import { uploadVideo } from '../services/api';

export default function UploadZone({ onUploadComplete }) {
    const [isDragOver, setIsDragOver] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [error, setError] = useState('');
    const fileInputRef = useRef(null);

    const allowedTypes = ['.mp4', '.mkv', '.avi', '.webm', '.mov'];

    const handleFile = async (file) => {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowedTypes.includes(ext)) {
            setError(`Unsupported format: ${ext}. Use: MP4, MKV, AVI, WebM, MOV`);
            return;
        }

        setError('');
        setUploading(true);
        setUploadProgress(0);

        try {
            const result = await uploadVideo(file, (progress) => {
                setUploadProgress(progress);
            });
            setUploading(false);
            setUploadProgress(100);
            if (onUploadComplete) onUploadComplete(result);
        } catch (err) {
            setUploading(false);
            setError(err.response?.data?.detail || 'Upload failed. Is the backend running?');
        }
    };

    const onDrop = (e) => {
        e.preventDefault();
        setIsDragOver(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    };

    const onDragOver = (e) => {
        e.preventDefault();
        setIsDragOver(true);
    };

    const onDragLeave = () => setIsDragOver(false);

    const onClick = () => fileInputRef.current?.click();

    const onFileSelect = (e) => {
        const file = e.target.files[0];
        if (file) handleFile(file);
    };

    return (
        <div
            id="upload-zone"
            className={`upload-zone ${isDragOver ? 'dragover' : ''}`}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={onClick}
        >
            <input
                ref={fileInputRef}
                type="file"
                accept=".mp4,.mkv,.avi,.webm,.mov"
                style={{ display: 'none' }}
                onChange={onFileSelect}
            />

            {!uploading ? (
                <>
                    <div className="upload-icon">🎬</div>
                    <div className="upload-title">Drop your video here</div>
                    <div className="upload-desc">
                        Supports MP4, MKV, AVI, WebM, MOV — up to 500MB
                    </div>
                    <button className="upload-btn" type="button">
                        Choose File
                    </button>
                </>
            ) : (
                <div className="progress-container">
                    <div className="upload-icon">⚡</div>
                    <div className="upload-title">Uploading...</div>
                    <div className="progress-bar-bg">
                        <div
                            className="progress-bar-fill"
                            style={{ width: `${uploadProgress}%` }}
                        />
                    </div>
                    <div className="progress-label">
                        <span>Uploading video</span>
                        <span>{uploadProgress}%</span>
                    </div>
                </div>
            )}

            {error && (
                <div style={{ color: 'var(--error)', marginTop: 12, fontSize: 13 }}>
                    ❌ {error}
                </div>
            )}
        </div>
    );
}
