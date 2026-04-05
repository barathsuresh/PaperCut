import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import ChatMessage from './ChatMessage'
import MessageInput from './MessageInput'

export default function ChatView({
  sessionName, modelType, messages, artifacts,
  isStreaming, onSend, onOpenFile, activeFile, onToggleCode,
}) {
  const bottomRef = useRef()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const totalFiles = (artifacts?.implementation_files?.length || 0) + (artifacts?.acceleration_files?.length || 0)

  return (
    <div className="chat-view">
      {/* Header */}
      <div className="chat-header">
        <div
          style={{
            width: 28, height: 28, borderRadius: 7, background: 'var(--accent)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 14, fontWeight: 700, color: '#fff', flexShrink: 0,
          }}
        >∂</div>
        <div className="chat-header-name">{sessionName || 'Paper'}</div>
        {modelType && <span className="chat-header-badge">{modelType}</span>}

        {totalFiles > 0 && (
          <button
            className={`files-toggle-btn${activeFile ? ' active' : ''}`}
            onClick={onToggleCode}
            title="Toggle generated files"
          >
            <svg width="13" height="13" viewBox="0 0 14 14" fill="none">
              <rect x="1" y="1" width="5" height="6" rx="1" stroke="currentColor" strokeWidth="1.4"/>
              <rect x="8" y="1" width="5" height="4" rx="1" stroke="currentColor" strokeWidth="1.4"/>
              <rect x="8" y="7" width="5" height="6" rx="1" stroke="currentColor" strokeWidth="1.4"/>
              <rect x="1" y="9" width="5" height="4" rx="1" stroke="currentColor" strokeWidth="1.4"/>
            </svg>
            Files
            <span className="files-toggle-count">{totalFiles}</span>
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 && !isStreaming && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{ textAlign: 'center', color: 'var(--text3)', fontSize: 14, marginTop: 40 }}
          >
            Ask anything about the paper or the generated code.
          </motion.div>
        )}
        {messages.map(m => (
          <ChatMessage key={m.id} role={m.role} text={m.text} streaming={m.streaming} statusText={m.statusText} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <MessageInput onSend={onSend} disabled={isStreaming} />
    </div>
  )
}
