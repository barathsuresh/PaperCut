import { motion } from 'framer-motion'

function timeAgo(iso) {
  if (!iso) return ''
  const diff = (Date.now() - new Date(iso).getTime()) / 1000
  if (diff < 60)  return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function sessionStatus(s) {
  if (s.node3_status === 'completed') return 'done'
  if (s.node2_status === 'completed') return 'done'
  if (s.node3_status === 'error' || s.node2_status === 'error') return 'done'
  if (s.scope_valid === false) return 'failed'
  if (s.scope_valid === true) return 'running'
  return 'pending'
}

function statusLabel(s) {
  if (s.node3_status === 'completed') return 'Ready'
  if (s.node2_status === 'completed') return 'Partial'
  if (s.scope_valid === false) return 'Rejected'
  if (s.scope_valid === true && !s.node2_status) return 'Processing'
  return 'Uploaded'
}

export default function Sidebar({ sessions, activeSessionId, onSelect, onNew, onDelete, mobileOpen, onMobileToggle }) {
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

      <div className="sidebar-section-label">Recent Papers</div>

      <div className="sidebar-sessions">
        {sessions.length === 0 && (
          <div style={{ padding: '16px 10px', color: 'var(--text3)', fontSize: 12, textAlign: 'center' }}>
            No papers yet
          </div>
        )}
        {sessions.map(s => {
          const st = sessionStatus(s)
          return (
            <motion.div
              key={s.session_id}
              className={`session-card${s.session_id === activeSessionId ? ' active' : ''}`}
              onClick={() => onSelect(s.session_id)}
              whileHover={{ x: 2 }}
              transition={{ duration: 0.1 }}
            >
              <div className="session-card-name">
                {s.name || 'Untitled Paper'}
              </div>
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
        })}
      </div>
    </aside>
  )
}
