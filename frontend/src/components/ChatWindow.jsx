import React, { useState, useRef, useEffect } from 'react';
import { chatWithVideo } from '../services/api';

function formatTimestamp(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

export default function ChatWindow({ videoId, onTimestampClick }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(scrollToBottom, [messages, loading]);

    // Reset chat when video changes
    useEffect(() => {
        setMessages([]);
    }, [videoId]);

    const sendMessage = async () => {
        if (!input.trim() || loading || !videoId) return;

        const userMsg = { role: 'user', content: input.trim() };
        setMessages((prev) => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const response = await chatWithVideo(
                userMsg.content,
                videoId,
                messages.map((m) => ({ role: m.role, content: m.content }))
            );

            const assistantMsg = {
                role: 'assistant',
                content: response.answer,
                timestamps: response.referenced_timestamps || [],
            };
            setMessages((prev) => [...prev, assistantMsg]);
        } catch (err) {
            const errorMsg = {
                role: 'assistant',
                content: '⚠️ Something went wrong. Make sure the backend is running and the video is processed.',
                timestamps: [],
            };
            setMessages((prev) => [...prev, errorMsg]);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const renderMessageContent = (msg) => {
        if (msg.role === 'user') return msg.content;

        // Render assistant message with clickable timestamp badges
        return (
            <div>
                <div>{msg.content}</div>
                {msg.timestamps && msg.timestamps.length > 0 && (
                    <div style={{ marginTop: 8, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                        {msg.timestamps.map((ts, i) => (
                            <span
                                key={i}
                                className="chat-timestamp-badge"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    if (onTimestampClick) onTimestampClick(ts);
                                }}
                            >
                                ⏱ {formatTimestamp(ts)}
                            </span>
                        ))}
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="chat-container">
            <div className="chat-header">
                🧠 Chat with Video
            </div>

            <div className="chat-messages">
                {messages.length === 0 && !loading && (
                    <div className="empty-state" style={{ padding: '40px 20px' }}>
                        <div className="empty-state-icon">💬</div>
                        <p>Ask anything about the video.</p>
                        <p style={{ fontSize: 12, marginTop: 8, color: 'var(--text-muted)' }}>
                            Try: "What is happening?" or "Show me when..."
                        </p>
                    </div>
                )}

                {messages.map((msg, i) => (
                    <div key={i} className={`chat-msg ${msg.role}`}>
                        {renderMessageContent(msg)}
                    </div>
                ))}

                {loading && (
                    <div className="typing-indicator">
                        <div className="typing-dot" />
                        <div className="typing-dot" />
                        <div className="typing-dot" />
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            <div className="chat-input-area">
                <textarea
                    id="chat-input"
                    className="chat-input"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={videoId ? "Ask about the video..." : "Upload a video first..."}
                    disabled={!videoId || loading}
                    rows={1}
                />
                <button
                    id="chat-send-btn"
                    className="chat-send-btn"
                    onClick={sendMessage}
                    disabled={!input.trim() || loading || !videoId}
                >
                    {loading ? <span className="spinner" /> : 'Send'}
                </button>
            </div>
        </div>
    );
}
