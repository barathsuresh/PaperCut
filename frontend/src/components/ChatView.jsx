import { useEffect, useRef, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ChatMessage from './ChatMessage'
import MessageInput from './MessageInput'

const STARTER_QUESTIONS = [
  'Explain the model architecture in simple terms.',
  'What is the key innovation compared to prior work?',
  'Walk me through the training process.',
  'How does the generated code relate to the paper?',
  'What are the main performance trade-offs?',
]

export default function ChatView({
  sessionName, modelType, messages, artifacts,
  isStreaming, onSend, onOpenFile, activeFile, onToggleCode,
  scopeReason, onNewAnalysis,
}) {
  const bottomRef   = useRef()
  const scrollRef   = useRef()
  const [showScrollBtn, setShowScrollBtn] = useState(false)

  const totalFiles = (artifacts?.implementation_files?.length || 0) + (artifacts?.acceleration_files?.length || 0)
  const isRejected = scopeReason != null

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  // Auto-scroll on new messages only when near bottom
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120
    if (isNearBottom) scrollToBottom()
  }, [messages, scrollToBottom])

  // Show/hide scroll-to-bottom button
  const onScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    setShowScrollBtn(el.scrollHeight - el.scrollTop - el.clientHeight > 180)
  }, [])

  return (
    <div className="chat-view">
      {/* Header */}
      <div className="chat-header">
        <div style={{
          width: 28, height: 28, borderRadius: 7, background: 'var(--accent)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 14, fontWeight: 700, color: '#fff', flexShrink: 0,
        }}>∂</div>
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
      <div className="chat-messages" ref={scrollRef} onScroll={onScroll}>
        {/* Rejected state */}
        {isRejected && (
          <motion.div
            className="rejected-card"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
          >
            <div className="rejected-icon">✕</div>
            <div className="rejected-title">Paper out of scope</div>
            <div className="rejected-reason">{scopeReason}</div>
            <button className="rejected-btn" onClick={onNewAnalysis}>
              Try another paper →
            </button>
          </motion.div>
        )}

        {/* Starter questions */}
        {!isRejected && messages.length === 0 && !isStreaming && (
          <motion.div
            className="starter-questions"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
          >
            <p className="starter-hint">Get started with a question:</p>
            {STARTER_QUESTIONS.map(q => (
              <button key={q} className="starter-q" onClick={() => onSend(q)}>
                {q}
              </button>
            ))}
          </motion.div>
        )}

        {messages.map(m => (
          <ChatMessage key={m.id} role={m.role} text={m.text} streaming={m.streaming} statusText={m.statusText} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Scroll to bottom */}
      <AnimatePresence>
        {showScrollBtn && (
          <motion.button
            className="scroll-to-bottom-btn"
            onClick={scrollToBottom}
            initial={{ opacity: 0, scale: 0.85 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.85 }}
            transition={{ duration: 0.15 }}
            title="Scroll to bottom"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M7 1v10M2 8l5 5 5-5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </motion.button>
        )}
      </AnimatePresence>

      {!isRejected && <MessageInput onSend={onSend} disabled={isStreaming} />}
    </div>
  )
}
