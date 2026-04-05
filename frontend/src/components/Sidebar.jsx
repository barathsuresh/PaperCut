import { motion } from 'framer-motion'
import { useState, useRef } from 'react'

function timeAgo(iso) {
  if (!iso) return ''
  const diff = (Date.now() - new Date(iso).getTime()) / 1000
  if (diff < 60)    return 'just now'
  if (diff < 3600)  return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function sessionStatus(s) {
  if (s.node3_status === 'completed') return 'done'
  if (s.node2_status === 'completed') return 'done'
  if (s.node3_status === 'error' || s.node2_status === 'error') return 'done'
  if (s.scope_valid === false) return 'failed'
  if (s.scope_valid === true)  return 'running'
  return 'pending'
}

function statusLabel(s) {
  if (s.node3_status === 'completed') return 'Ready'
  if (s.node2_status === 'completed') return 'Partial'
  if (s.scope_valid === false)        return 'Rejected'
  if (s.scope_valid === true && !s.node2_status) return 'Processing'
  return 'Uploaded'
}

function SessionCard({ s, isActive, onSelect, onDelete, onRename }) {
  const [editing, setEditing]   = useState(false)
  const [draft,   setDraft]     = useState('')
  const inputRef = useRef()

  const startEdit = e => {
    e.stopPropagation()
    setDraft(s.name || 'Untitled Paper')
    setEditing(true)
    setTimeout(() => inputRef.current?.select(), 0)
  }

  const commitRename = () => {
    setEditing(false)
    const trimmed = draft.trim()
    if (trimmed && trimmed !== s.name) onRename(s.session_id, trimmed)
  }

  const onKeyDown = e => {
    if (e.key === 'Enter')  { e.preventDefault(); commitRename() }
    if (e.key === 'Escape') { setEditing(false) }
  }

  const st = sessionStatus(s)

  return (
    <motion.div
      className={`session-card${isActive ? ' active' : ''}`}
      onClick={() => !editing && onSelect(s.session_id)}
      whileHover={{ x: 2 }}
      transition={{ duration: 0.1 }}
    >
      {editing ? (
        <input
          ref={inputRef}
          className="session-rename-input"
          value={draft}
          onChange={e => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          onBlur={commitRename}
          onClick={e => e.stopPropagation()}
          autoFocus
        />
      ) : (
        <div className="session-card-name" onDoubleClick={startEdit} title="Double-click to rename">
          {s.name || 'Untitled Paper'}
        </div>
      )}

      <div className="session-card-meta">
        <span className={`status-dot ${st}`} />
        <span>{statusLabel(s)}</span>
        {s.uploaded_at && <span>· {timeAgo(s.uploaded_at)}</span>}
      </div>

      <button
        className="session-delete-btn"
        title="Delete session"
        onClick={e => { e.stopPropagation(); onDelete(s.session_id) }}
      >
        <svg width="11" height="11" viewBox="0 0 12 12" fill="none">
          <path d="M1 1l10 10M11 1L1 11" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
        </svg>
      </button>
    </motion.div>
  )
}

export default function Sidebar({ sessions, activeSessionId, onSelect, onNew, onDelete, onRename, mobileOpen }) {
  const [search, setSearch] = useState('')

  const filtered = search.trim()
    ? sessions.filter(s => (s.name || '').toLowerCase().includes(search.toLowerCase()))
    : sessions

  return (
    <aside className={`sidebar${mobileOpen ? ' mobile-open' : ''}`}>
      <div className="sidebar-header">
        <div className="sidebar-logo" style={{ fontSize: 17, letterSpacing: 0 }}>∂</div>
        <span className="sidebar-brand">PaperCut</span>
      </div>

      <button className="sidebar-new-btn" onClick={onNew}>
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M7 1v12M1 7h12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
        </svg>
        New Analysis
      </button>

      {sessions.length > 3 && (
        <div className="sidebar-search-wrap">
          <svg width="12" height="12" viewBox="0 0 14 14" fill="none" className="sidebar-search-icon">
            <circle cx="6" cy="6" r="4.5" stroke="currentColor" strokeWidth="1.4"/>
            <path d="M9.5 9.5l3 3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
          </svg>
          <input
            className="sidebar-search"
            placeholder="Search papers…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
      )}

      <div className="sidebar-section-label">Recent Papers</div>

      <div className="sidebar-sessions">
        {filtered.length === 0 && (
          <div style={{ padding: '16px 10px', color: 'var(--text3)', fontSize: 12, textAlign: 'center' }}>
            {search ? 'No matches' : 'No papers yet'}
          </div>
        )}
        {filtered.map(s => (
          <SessionCard
            key={s.session_id}
            s={s}
            isActive={s.session_id === activeSessionId}
            onSelect={onSelect}
            onDelete={onDelete}
            onRename={onRename}
          />
        ))}
      </div>
    </aside>
  )
}
