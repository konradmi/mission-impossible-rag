'use client';

import React, { useState } from 'react';
import { useChat } from '@ai-sdk/react';

export default function ChatPage() {
  const [input, setInput] = useState('');
  const { messages, append } = useChat({ api: '/api/prompt' });
  console.log(messages);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    await append({ role: 'user', content: input });
    setInput('');
  };

  return (
    <div style={{ maxWidth: 500, margin: '40px auto', padding: 16, border: '1px solid #eee', borderRadius: 8 }}>
      <h2>Mission Impossible RAG Chat</h2>
      <div style={{ minHeight: 200, marginBottom: 16 }}>
        {messages.map((msg, idx) => (
          <div key={idx} style={{ margin: '8px 0', textAlign: msg.role === 'user' ? 'right' : 'left' }}>
            <span style={{ fontWeight: msg.role === 'user' ? 'bold' : 'normal' }}>
              {msg.role === 'user' ? 'You' : 'AI'}:
            </span>{' '}
            {msg.content}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: 8 }}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Type your message..."
          style={{ flex: 1, padding: 8, borderRadius: 4, border: '1px solid #ccc' }}
        />
        <button type="submit" style={{ padding: '8px 16px', borderRadius: 4, border: 'none', background: '#222', color: '#fff' }}>
          Send
        </button>
      </form>
    </div>
  );
} 
