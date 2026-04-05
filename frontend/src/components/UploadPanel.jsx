import { useState, useRef } from 'react'
import { motion } from 'framer-motion'

export default function UploadPanel({ onSubmit, loading }) {
  const [tab, setTab] = useState('url')
  const [url, setUrl]     = useState('')
  const [file, setFile]   = useState(null)
  const [drag, setDrag]   = useState(false)
  const [error, setError] = useState('')
  const inputRef = useRef()

  const handleDrop = e => {
    e.preventDefault(); setDrag(false)
    const f = e.dataTransfer.files[0]
    if (f && f.type === 'application/pdf') { setFile(f); setError('') }
    else setError('Please drop a PDF file.')
  }

  const handleFileInput = e => {
    const f = e.target.files[0]
    if (f) { setFile(f); setError('') }
  }

  const handleSubmit = async () => {
    setError('')
    if (tab === 'url' && !url.trim()) { setError('Please enter a URL.'); return }
    if (tab === 'file' && !file)      { setError('Please select a PDF.'); return }

    const fd = new FormData()
    if (tab === 'url') fd.append('pdf_url', url.trim())
    else               fd.append('file', file)

    try { await onSubmit(fd) }
    catch (e) { setError(e.message) }
  }

  return (
    <div className="upload-panel">
      <motion.div
        className="upload-card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
      >
        <div className="upload-card-title">Analyse a paper</div>
        <div className="upload-card-sub">
          Paste an ArXiv URL or upload a PDF — we'll extract the architecture and generate optimised code.
        </div>

        <div className="upload-tabs">
          <button className={`upload-tab${tab === 'url'  ? ' active' : ''}`} onClick={() => setTab('url')}>URL</button>
          <button className={`upload-tab${tab === 'file' ? ' active' : ''}`} onClick={() => setTab('file')}>Upload PDF</button>
        </div>

        {tab === 'url' ? (
          <input
            className="upload-input"
            type="url"
            placeholder="https://arxiv.org/pdf/2307.09288"
            value={url}
            onChange={e => { setUrl(e.target.value); setError('') }}
            onKeyDown={e => e.key === 'Enter' && !loading && handleSubmit()}
            autoFocus
          />
        ) : (
          <>
            <div
              className={`upload-drop${drag ? ' drag-over' : ''}`}
              onDragOver={e => { e.preventDefault(); setDrag(true) }}
              onDragLeave={() => setDrag(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
            >
              <div className="upload-drop-icon">📄</div>
              {file
                ? <div className="upload-file-name">{file.name}</div>
                : <div>Drop a PDF here or <span style={{ color: 'var(--accent2)' }}>click to browse</span></div>
              }
            </div>
            <input
              ref={inputRef} type="file" accept="application/pdf"
              style={{ display: 'none' }} onChange={handleFileInput}
            />
          </>
        )}

        {error && <div className="upload-error">{error}</div>}

        <button className="upload-submit" onClick={handleSubmit} disabled={loading}>
          {loading ? <><span className="loader loader-sm" /> Uploading…</> : 'Analyse Paper →'}
        </button>
      </motion.div>
    </div>
  )
}
