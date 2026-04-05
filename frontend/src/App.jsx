import { useState, useEffect, useCallback, useRef } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import SplashScreen     from './components/SplashScreen'
import Sidebar          from './components/Sidebar'
import UploadPanel      from './components/UploadPanel'
import PipelineProgress from './components/PipelineProgress'
import ChatView         from './components/ChatView'
import CodePanel        from './components/CodePanel'
import ToastContainer   from './components/Toast'
import useToast         from './hooks/useToast'
import { getSessions, getSession, getHistory, getArtifacts, getArtifactContent, uploadPdf, streamPipeline, streamChat, deleteSession, renameSession } from './api/client'

let msgId = 0
const newId = () => `m${++msgId}`

export default function App() {
  const { toasts, toast, remove: removeToast } = useToast()
  const [splashDone,   setSplashDone]   = useState(false)
  const [sessions,     setSessions]     = useState([])
  const [activeId,     setActiveId]     = useState(null)
  const [activeDetail, setActiveDetail] = useState(null)
  const [view,         setView]         = useState('home')
  const [pipeEvents,   setPipeEvents]   = useState([])
  const [messages,     setMessages]     = useState([])
  const [artifacts,    setArtifacts]    = useState(null)
  const [codeOpen,     setCodeOpen]     = useState(false)
  const [activeFile,   setActiveFile]   = useState(null)
  const [fileContent,  setFileContent]  = useState(null)
  const [fileError,    setFileError]    = useState(null)
  const [loadingFile,  setLoadingFile]  = useState(false)
  const [uploading,    setUploading]    = useState(false)
  const [isStreaming,  setIsStreaming]  = useState(false)
  const [mobileOpen,   setMobileOpen]  = useState(false)
  const stopPipe = useRef(null)

  /* ── Load session list ── */
  const loadSessions = useCallback(async () => {
    try { setSessions(await getSessions()) } catch {}
  }, [])

  useEffect(() => { if (splashDone) loadSessions() }, [splashDone, loadSessions])

  /* ── Dynamic page title ── */
  useEffect(() => {
    const name = activeDetail?.name || sessions.find(s => s.session_id === activeId)?.name
    document.title = name ? `∂ ${name} — PaperCut` : '∂ PaperCut'
  }, [activeId, activeDetail, sessions])

  /* ── Global drag-and-drop ── */
  useEffect(() => {
    const over = e => { e.preventDefault() }
    const drop = e => {
      e.preventDefault()
      const f = e.dataTransfer.files[0]
      if (f && f.type === 'application/pdf') {
        setView('upload')
        setActiveId(null); setActiveDetail(null)
        setPipeEvents([]); setMessages([]); setArtifacts(null)
        setCodeOpen(false); setActiveFile(null); setFileContent(null); setFileError(null)
        setTimeout(async () => {
          const fd = new FormData(); fd.append('file', f)
          try { await handleUpload(fd) } catch {}
        }, 100)
      }
    }
    window.addEventListener('dragover', over)
    window.addEventListener('drop', drop)
    return () => { window.removeEventListener('dragover', over); window.removeEventListener('drop', drop) }
  }, []) // eslint-disable-line

  /* ── Select session ── */
  const handleSelectSession = useCallback(async id => {
    if (stopPipe.current) { stopPipe.current(); stopPipe.current = null }
    setActiveId(id)
    setMobileOpen(false)
    setPipeEvents([])
    setMessages([])
    setArtifacts(null)
    setCodeOpen(false)
    setActiveFile(null)
    setFileContent(null)
    setFileError(null)

    try {
      const [detail, history, arts] = await Promise.all([
        getSession(id),
        getHistory(id),
        getArtifacts(id),
      ])
      setActiveDetail(detail)
      setArtifacts(arts)
      const msgs = history.map(h => ({ id: newId(), role: h.role, text: h.text, streaming: false }))
      setMessages(msgs)
      setView('chat')
    } catch {
      setView('home')
    }
  }, [])

  /* ── Delete session ── */
  const handleDelete = useCallback(async id => {
    try { await deleteSession(id) } catch {}
    setSessions(prev => prev.filter(s => s.session_id !== id))
    toast('Session deleted', 'success')
    if (activeId === id) {
      if (stopPipe.current) { stopPipe.current(); stopPipe.current = null }
      setActiveId(null); setActiveDetail(null); setView('home')
      setPipeEvents([]); setMessages([]); setArtifacts(null)
      setCodeOpen(false); setActiveFile(null); setFileContent(null); setFileError(null)
    }
  }, [activeId, toast])

  /* ── Rename session ── */
  const handleRename = useCallback(async (id, name) => {
    try {
      await renameSession(id, name)
      setSessions(prev => prev.map(s => s.session_id === id ? { ...s, name } : s))
      if (id === activeId) setActiveDetail(prev => prev ? { ...prev, name } : prev)
      toast('Renamed', 'success', 1800)
    } catch {
      toast('Rename failed', 'error')
    }
  }, [activeId, toast])

  /* ── New session ── */
  const handleNew = () => {
    if (stopPipe.current) { stopPipe.current(); stopPipe.current = null }
    setActiveId(null); setActiveDetail(null); setView('upload')
    setPipeEvents([]); setMessages([]); setArtifacts(null)
    setCodeOpen(false); setActiveFile(null); setFileContent(null); setFileError(null)
    setMobileOpen(false)
  }

  /* ── Upload + auto-start pipeline ── */
  const handleUpload = async formData => {
    setUploading(true)
    try {
      const { session_id } = await uploadPdf(formData)
      setActiveId(session_id)
      setActiveDetail(null)
      setPipeEvents([])
      setView('pipeline')
      await loadSessions()

      stopPipe.current = streamPipeline(session_id, ev => {
        setPipeEvents(prev => [...prev, ev])
        if (ev.type === 'done') {
          loadSessions()
          getSession(session_id).then(d => setActiveDetail(d)).catch(() => {})
          getArtifacts(session_id).then(a => setArtifacts(a)).catch(() => {})
          if (ev.scope_valid === false) {
            setView('chat')
          } else {
            setTimeout(() => setView('chat'), 1400)
          }
          stopPipe.current = null
        }
        if (ev.type === 'error') {
          loadSessions()
          stopPipe.current = null
        }
      })
    } finally {
      setUploading(false)
    }
  }

  /* ── Send chat message ── */
  const handleSend = useCallback(text => {
    if (isStreaming || !activeId) return
    const userMsg = { id: newId(), role: 'user', text, streaming: false }
    const asstMsg = { id: newId(), role: 'assistant', text: '', streaming: true }
    setMessages(prev => [...prev, userMsg, asstMsg])
    setIsStreaming(true)

    streamChat(activeId, text, ev => {
      if (ev.type === 'status') {
        setMessages(prev => prev.map(m =>
          m.id === asstMsg.id ? { ...m, statusText: ev.text } : m
        ))
      }
      if (ev.type === 'token') {
        setMessages(prev => prev.map(m =>
          m.id === asstMsg.id ? { ...m, statusText: null, text: m.text + ev.text } : m
        ))
      }
      if (ev.type === 'done' || ev.type === 'error') {
        setMessages(prev => prev.map(m =>
          m.id === asstMsg.id ? { ...m, statusText: null, streaming: false } : m
        ))
        setIsStreaming(false)
      }
    })
  }, [activeId, isStreaming])

  /* ── Open / switch code file ── */
  const loadFile = useCallback(async (group, fileName) => {
    if (activeFile?.group === group && activeFile?.fileName === fileName) return
    setActiveFile({ group, fileName })
    setLoadingFile(true)
    setFileContent(null)
    setFileError(null)
    try {
      const content = await getArtifactContent(activeId, group, fileName)
      setFileContent(content)
    } catch (e) {
      setFileError(e.message || 'Could not load file.')
    } finally {
      setLoadingFile(false)
    }
  }, [activeId, activeFile])

  const handleOpenFile  = useCallback(async (group, fileName) => {
    setCodeOpen(true)
    await loadFile(group, fileName)
  }, [loadFile])

  const handleSelectFile = useCallback(async (group, fileName) => {
    await loadFile(group, fileName)
  }, [loadFile])

  const sessionName = activeDetail?.name || sessions.find(s => s.session_id === activeId)?.name || ''
  const modelType   = activeDetail?.model_type || sessions.find(s => s.session_id === activeId)?.model_type || ''
  const scopeReason = activeDetail?.scope_valid === false
    ? (activeDetail?.scope_reason || 'Paper did not pass scope check.')
    : null

  return (
    <>
      <AnimatePresence>
        {!splashDone && <SplashScreen onComplete={() => setSplashDone(true)} />}
      </AnimatePresence>

      {splashDone && (
        <div className="app-layout">
          <div className="mobile-topbar">
            <button className="mobile-menu-btn" onClick={() => setMobileOpen(o => !o)}>☰</button>
            <span style={{ fontWeight: 700, fontSize: 14 }}>∂ PaperCut</span>
          </div>

          <Sidebar
            sessions={sessions}
            activeSessionId={activeId}
            onSelect={handleSelectSession}
            onNew={handleNew}
            onDelete={handleDelete}
            onRename={handleRename}
            mobileOpen={mobileOpen}
          />

          {mobileOpen && (
            <div
              onClick={() => setMobileOpen(false)}
              style={{ position: 'fixed', inset: 0, zIndex: 40, background: 'rgba(0,0,0,0.6)' }}
            />
          )}

          <main className="main-content">
            <AnimatePresence mode="wait">
              {view === 'home' && (
                <motion.div key="home" className="home-view"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <div className="home-logo" style={{ fontSize: 36, letterSpacing: 0 }}>∂</div>
                  <div className="home-title">Cut through the paper</div>
                  <div className="home-sub">
                    Upload an ML research paper and PaperCut will analyse its architecture,
                    generate implementation code, and let you explore it through chat.
                  </div>
                  <button className="home-btn" onClick={handleNew}>New Analysis →</button>
                </motion.div>
              )}

              {view === 'upload' && (
                <motion.div key="upload" style={{ display: 'flex', flex: 1, overflow: 'hidden' }}
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <UploadPanel onSubmit={handleUpload} loading={uploading} />
                </motion.div>
              )}

              {view === 'pipeline' && (
                <motion.div key="pipeline" style={{ display: 'flex', flex: 1, overflow: 'hidden' }}
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <PipelineProgress events={pipeEvents} sessionName={sessionName} />
                </motion.div>
              )}

              {(view === 'chat' || view === 'rejected') && (
                <motion.div key="chat" style={{ display: 'flex', flex: 1, overflow: 'hidden' }}
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  transition={{ duration: 0.25 }}
                >
                  <ChatView
                    sessionName={sessionName}
                    modelType={modelType}
                    messages={messages}
                    artifacts={artifacts}
                    isStreaming={isStreaming}
                    onSend={handleSend}
                    onOpenFile={handleOpenFile}
                    activeFile={codeOpen ? activeFile : null}
                    scopeReason={scopeReason}
                    onNewAnalysis={handleNew}
                    onToggleCode={() => {
                      if (codeOpen) {
                        setCodeOpen(false); setActiveFile(null); setFileContent(null); setFileError(null)
                      } else {
                        const firstGroup = artifacts?.implementation_files?.length ? 'implementation' : 'acceleration'
                        const firstFile  = artifacts?.[`${firstGroup}_files`]?.[0]
                        if (firstFile) handleOpenFile(firstGroup, firstFile)
                        else setCodeOpen(true)
                      }
                    }}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </main>

          <AnimatePresence>
            {codeOpen && (
              <CodePanel
                key="codepanel"
                open={codeOpen}
                artifacts={artifacts}
                activeFile={activeFile}
                onSelectFile={handleSelectFile}
                onClose={() => { setCodeOpen(false); setActiveFile(null); setFileContent(null); setFileError(null) }}
                loadingFile={loadingFile}
                fileContent={fileContent}
                fileError={fileError}
              />
            )}
          </AnimatePresence>
        </div>
      )}

      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </>
  )
}
