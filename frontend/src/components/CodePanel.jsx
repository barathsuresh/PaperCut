import { motion } from 'framer-motion'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { useState, useRef, useCallback } from 'react'

const MIN_WIDTH = 320
const MAX_WIDTH = Math.round(window.innerWidth * 0.75)

const codeStyle = {
  ...vscDarkPlus,
  'pre[class*="language-"]': {
    ...vscDarkPlus['pre[class*="language-"]'],
    background: 'transparent',
    margin: 0, padding: '16px',
    fontSize: '12.5px', lineHeight: '1.6',
    fontFamily: "'JetBrains Mono', monospace",
  },
  'code[class*="language-"]': {
    ...vscDarkPlus['code[class*="language-"]'],
    background: 'none',
    fontSize: '12.5px', fontFamily: "'JetBrains Mono', monospace",
  },
}

const LANG_MAP = {
  '.py': 'python', '.yaml': 'yaml', '.yml': 'yaml',
  '.json': 'json', '.cu': 'cpp', '.cuh': 'cpp',
  '.cpp': 'cpp', '.md': 'markdown', '.txt': 'text',
}

function ext(name) { return name.slice(name.lastIndexOf('.')) }

const GROUP_CONFIG = {
  implementation: { label: 'Implementation', color: '#60a5fa' },
  acceleration:   { label: 'Accelerators',   color: '#34d399' },
}


export default function CodePanel({
  open, artifacts, activeFile, onSelectFile, onClose,
  loadingFile, fileContent, fileError,
}) {
  const [copied,   setCopied]   = useState(false)
  const [wrap,     setWrap]     = useState(false)
  const [fontSize, setFontSize] = useState(12.5)
  const [width,    setWidth]    = useState(560)
  const dragging = useRef(false)
  const startX   = useRef(0)
  const startW   = useRef(0)

  const onDragStart = useCallback(e => {
    dragging.current = true
    startX.current = e.clientX
    startW.current = width
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    const onMove = ev => {
      if (!dragging.current) return
      const delta = startX.current - ev.clientX
      setWidth(Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, startW.current + delta)))
    }
    const onUp = () => {
      dragging.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [width])

  if (!open) return null

  const handleCopy = () => {
    if (!fileContent?.content) return
    navigator.clipboard.writeText(fileContent.content).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    })
  }

  return (
    <motion.aside
      className="code-panel"
      style={{ width, minWidth: width }}
      initial={{ x: '100%', opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: '100%', opacity: 0 }}
      transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
    >
      {/* Resize handle */}
      <div className="code-panel-resize-handle" onMouseDown={onDragStart} />

      {/* File tree */}
      <div className="code-tree">
        <div className="code-tree-header">
          <span>Explorer</span>
          <button className="code-panel-close" onClick={onClose} title="Close">✕</button>
        </div>

        {['implementation', 'acceleration'].map(group => {
          const files = group === 'implementation'
            ? artifacts?.implementation_files
            : artifacts?.acceleration_files
          if (!files?.length) return null
          const { label, color } = GROUP_CONFIG[group]
          return (
            <div key={group} className="code-tree-group">
              <div className="code-tree-group-label" style={{ color }}>
                {label.toUpperCase()}
              </div>
              {files.map(f => {
                const isActive = activeFile?.fileName === f && activeFile?.group === group
                return (
                  <button
                    key={f}
                    className={`code-tree-file${isActive ? ' active' : ''}`}
                    onClick={() => onSelectFile(group, f)}
                    style={isActive ? { color: '#fff', background: 'rgba(255,255,255,0.07)' } : { color }}
                    title={f}
                  >
                    <span className="code-tree-file-icon">
                      {ext(f) === '.py' ? '𝜆' : ext(f).includes('cu') ? '◈' : '◻'}
                    </span>
                    <span className="code-tree-file-name">{f}</span>
                  </button>
                )
              })}
            </div>
          )
        })}

        {!artifacts?.implementation_files?.length && !artifacts?.acceleration_files?.length && (
          <div className="code-tree-empty">No files yet</div>
        )}
      </div>

      {/* Code area */}
      <div className="code-panel-body">
        {loadingFile && (
          <div className="code-panel-center">
            <span className="loader" />
          </div>
        )}

        {!loadingFile && fileError && (
          <div className="code-panel-center" style={{ flexDirection: 'column', gap: 12 }}>
            <span style={{ fontSize: 26, opacity: 0.4 }}>⚠</span>
            <span style={{ color: 'var(--red)', fontSize: 13 }}>Could not load file</span>
            <span style={{ color: 'var(--text3)', fontSize: 11, maxWidth: 200, textAlign: 'center' }}>{fileError}</span>
          </div>
        )}

        {!loadingFile && !fileError && !fileContent && (
          <div className="code-panel-center" style={{ flexDirection: 'column', gap: 10 }}>
            <span style={{ fontSize: 32 }}>{'</>'}</span>
            <span style={{ fontSize: 13 }}>Select a file from the tree</span>
          </div>
        )}

        {!loadingFile && fileContent && (
          <div style={{ position: 'relative', height: '100%', overflow: 'auto' }}>
            <div className="code-panel-topbar">
              <span className="code-panel-filename">{fileContent.file_name}</span>
              <div className="code-panel-controls">
                <button
                  className={`code-ctrl-btn${wrap ? ' active' : ''}`}
                  onClick={() => setWrap(w => !w)}
                  title="Toggle line wrap"
                >⏎</button>
                <button
                  className="code-ctrl-btn"
                  onClick={() => setFontSize(s => Math.max(10, s - 1))}
                  title="Decrease font size"
                >A-</button>
                <button
                  className="code-ctrl-btn"
                  onClick={() => setFontSize(s => Math.min(18, s + 1))}
                  title="Increase font size"
                >A+</button>
                <button className="code-panel-copy-btn" onClick={handleCopy}>
                  {copied ? '✓ Copied' : 'Copy'}
                </button>
              </div>
            </div>
            <SyntaxHighlighter
              language={LANG_MAP[ext(fileContent.file_name)] || 'text'}
              style={{
                ...codeStyle,
                'pre[class*="language-"]': {
                  ...codeStyle['pre[class*="language-"]'],
                  fontSize: `${fontSize}px`,
                },
                'code[class*="language-"]': {
                  ...codeStyle['code[class*="language-"]'],
                  fontSize: `${fontSize}px`,
                },
              }}
              showLineNumbers
              lineNumberStyle={{ color: 'var(--text3)', fontSize: '11px', minWidth: '2.5em' }}
              wrapLongLines={wrap}
            >
              {fileContent.content}
            </SyntaxHighlighter>
          </div>
        )}
      </div>
    </motion.aside>
  )
}
