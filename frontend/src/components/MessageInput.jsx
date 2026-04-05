import { useRef, useState } from 'react'

export default function MessageInput({ onSend, disabled }) {
  const [text, setText] = useState('')
  const ref = useRef()

  const send = () => {
    const msg = text.trim()
    if (!msg || disabled) return
    setText('')
    ref.current.style.height = 'auto'
    onSend(msg)
  }

  const onKey = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  const onInput = e => {
    setText(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
  }

  return (
    <div className="msg-input-area">
      <div className="msg-input-row">
        <textarea
          ref={ref}
          className="msg-input-textarea"
          rows={1}
          placeholder="Ask anything about this paper…"
          value={text}
          onChange={onInput}
          onKeyDown={onKey}
          disabled={disabled}
        />
        <button className="msg-send-btn" onClick={send} disabled={disabled || !text.trim()} title="Send (Enter)">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M1 7h12M7 1l6 6-6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>
    </div>
  )
}
