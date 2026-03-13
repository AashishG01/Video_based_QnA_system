/**
 * API Service Layer — All backend communication
 */
import axios from 'axios';

const api = axios.create({
    baseURL: '/api',
    timeout: 120000, // 2 min timeout for VLM inference
});

// =====================================================
// VIDEO UPLOAD
// =====================================================
export async function uploadVideo(file, onProgress) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/videos/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (e) => {
            if (onProgress && e.total) {
                onProgress(Math.round((e.loaded / e.total) * 100));
            }
        },
    });
    return response.data;
}

// =====================================================
// VIDEO STATUS & LISTING
// =====================================================
export async function getVideoStatus(videoId) {
    const response = await api.get(`/videos/${videoId}/status`);
    return response.data;
}

export async function listVideos() {
    const response = await api.get('/videos');
    return response.data;
}

// =====================================================
// TRANSCRIPT
// =====================================================
export async function getTranscript(videoId) {
    const response = await api.get(`/videos/${videoId}/transcript`);
    return response.data;
}

// =====================================================
// SEARCH
// =====================================================
export async function searchVideo(query, videoId, searchType = 'hybrid', limit = 5) {
    const response = await api.post('/search', {
        query,
        video_id: videoId,
        search_type: searchType,
        limit,
    });
    return response.data;
}

// =====================================================
// CHAT
// =====================================================
export async function chatWithVideo(question, videoId, chatHistory = []) {
    const response = await api.post('/chat', {
        question,
        video_id: videoId,
        chat_history: chatHistory,
    });
    return response.data;
}

export default api;
