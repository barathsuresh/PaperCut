import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { motion, AnimatePresence } from 'framer-motion'
import { useState, useEffect } from 'react'

const codeStyle = {
  ...vscDarkPlus,
  'pre[class*="language-"]': {
    ...vscDarkPlus['pre[class*="language-"]'],
    background: 'var(--surface2)',
    margin: 0, padding: '14px',
    borderRadius: '8px', fontSize: '12.5px',
    lineHeight: '1.6', fontFamily: "'JetBrains Mono', monospace",
  },
  'code[class*="language-"]': {
    ...vscDarkPlus['code[class*="language-"]'],
    background: 'none', fontSize: '12.5px',
    fontFamily: "'JetBrains Mono', monospace",
  },
}

function AnimatedDots() {
  const [dots, setDots] = useState('.')
  useEffect(() => {
    const t = setInterval(() => setDots(d => d.length >= 3 ? '.' : d + '.'), 380)
    return () => clearInterval(t)
  }, [])
  return (
    <span style={{ display: 'inline-block', minWidth: '1.4em', color: 'var(--accent)' }}>
      {dots}
    </span>
  )
}

const components = {
  code({ node, inline, className, children, ...props }) {
    const lang = /language-(\w+)/.exec(className || '')?.[1]
    if (!inline && lang) {
      return (
        <SyntaxHighlighter language={lang} style={codeStyle} PreTag="div" {...props}>
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      )
    }
    return <code className={className} {...props}>{children}</code>
  },
}

export default function ChatMessage({ role, text, streaming, statusText }) {
  const isUser = role === 'user'
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    if (!text) return
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    })
  }

  return (
    <motion.div
      className={`chat-message ${role}`}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22 }}
    >
      <div className={`msg-avatar ${role}`}>
        {isUser ? 'U' : '∂'}
      </div>

      <div className="msg-bubble-wrap">
        <div className="msg-bubble">
          {statusText ? (
            <AnimatePresence mode="popLayout">
              <motion.span
                key={statusText}
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.18 }}
                style={{ color: 'var(--text3)', fontStyle: 'italic', fontSize: 13, display: 'inline-flex', alignItems: 'baseline', gap: 1 }}
              >
                {statusText.replace(/\.+$/, '')}
                <AnimatedDots />
              </motion.span>
            </AnimatePresence>
          ) : streaming ? (
            <>
              <span style={{ whiteSpace: 'pre-wrap' }}>{text}</span>
              <span style={{ animation: 'blink 0.85s ease-in-out infinite', color: 'var(--accent2)' }}>_</span>
            </>
          ) : isUser ? (
            <span style={{ whiteSpace: 'pre-wrap' }}>{text}</span>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
              {text}
            </ReactMarkdown>
          )}
        </div>

        {/* Copy button — only for completed assistant messages */}
        {!isUser && !streaming && !statusText && text && (
          <button className="msg-copy-btn" onClick={handleCopy} title="Copy message">
            {copied
              ? <svg width="13" height="13" viewBox="0 0 14 14" fill="none"><path d="M2 7l3.5 3.5L12 3" stroke="var(--green)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>
              : <svg width="13" height="13" viewBox="0 0 14 14" fill="none"><rect x="5" y="1" width="8" height="10" rx="1.5" stroke="currentColor" strokeWidth="1.3"/><path d="M1 4v8.5A1.5 1.5 0 002.5 14H9" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>
            }
          </button>
        )}
      </div>
    </motion.div>
  )
}
