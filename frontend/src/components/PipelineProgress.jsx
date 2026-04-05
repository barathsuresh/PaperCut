import { motion, AnimatePresence } from 'framer-motion'
import { useState } from 'react'

const STEPS = [
  { key: 'node0', label: 'Reading Paper',          runDesc: 'Checking scope and relevance…',      hint: '~10s'  },
  { key: 'node1', label: 'Analysing Architecture',  runDesc: 'Extracting model blueprint…',        hint: '~30s'  },
  { key: 'node2', label: 'Building Implementation', runDesc: 'Generating PyTorch code…',           hint: '~1 min' },
  { key: 'node3', label: 'Designing Accelerators',  runDesc: 'Creating CUDA optimisation stubs…', hint: '~2 min' },
]

function stepSummary(key, ev) {
  if (!ev) return ''
  switch (key) {
    case 'node0': return ev.result === 'PASS' ? 'Paper is in scope' : 'Out of scope'
    case 'node1': return ev.model_type ? `Architecture identified: ${ev.model_type}` : 'Blueprint ready'
    case 'node2': {
      const n = Array.isArray(ev.files) ? ev.files.length : Object.keys(ev.files || {}).length
      return n ? `${n} implementation file${n > 1 ? 's' : ''} generated` : 'Implementation ready'
    }
    case 'node3': {
      const s = ev.stub_files?.length ?? 0
      const b = ev.bottleneck_count ?? ev.bottlenecks?.length ?? 0
      return `${s} CUDA stub${s !== 1 ? 's' : ''}, ${b} bottleneck${b !== 1 ? 's' : ''} identified`
    }
    default: return ev.message || 'Done'
  }
}

function stepDetail(key, ev) {
  if (!ev) return null
  switch (key) {
    case 'node0': return ev.message || null
    case 'node1': return ev.message || null
    case 'node2': {
      const files = Array.isArray(ev.files) ? ev.files : Object.keys(ev.files || {})
      return files.length ? files.join(', ') : null
    }
    case 'node3': {
      const stubs = ev.stub_files || []
      return stubs.length ? stubs.join(', ') : null
    }
    default: return null
  }
}

function getStepStates(events) {
  const states  = { node0: 'pending', node1: 'pending', node2: 'pending', node3: 'pending' }
  const details = {}
  for (const ev of events) {
    if (ev.type === 'node_start') states[ev.node]  = 'running'
    if (ev.type === 'node_done')  { states[ev.node] = 'step-done'; details[ev.node] = ev }
    if (ev.type === 'node_error') { states[ev.node] = 'error';     details[ev.node] = ev }
  }
  return { states, details }
}

function StepIcon({ status }) {
  if (status === 'running')   return <span className="spin" style={{ fontSize: 13, display: 'inline-block' }}>⟳</span>
  if (status === 'step-done') return <span>✓</span>
  if (status === 'error')     return <span>✕</span>
  return null
}

export default function PipelineProgress({ events, sessionName }) {
  const { states, details } = getStepStates(events)
  const [expanded, setExpanded] = useState({})
  const doneEvent  = events.find(e => e.type === 'done')
  const errorEvent = events.find(e => e.type === 'error')
  const rejected   = doneEvent && doneEvent.scope_valid === false

  const toggle = key => setExpanded(prev => ({ ...prev, [key]: !prev[key] }))

  return (
    <div className="pipeline-view">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        style={{ width: '100%', maxWidth: 440, display: 'flex', flexDirection: 'column', alignItems: 'center' }}
      >
        <div className="pipeline-title">{sessionName || 'Analysing paper…'}</div>
        <div className="pipeline-subtitle">
          {doneEvent && !rejected ? 'Analysis complete' : 'Usually takes 2–3 minutes end to end'}
        </div>

        <div className="pipeline-steps">
          {STEPS.map((step, i) => {
            const st      = states[step.key]
            const det     = details[step.key]
            const summary = stepSummary(step.key, det)
            const detail  = stepDetail(step.key, det)
            const isOpen  = expanded[step.key]
            const canExpand = (st === 'step-done' || st === 'error') && detail

            return (
              <motion.div
                key={step.key}
                className={`pipeline-step${st === 'step-done' ? ' step-done' : ''}`}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.07, duration: 0.35 }}
              >
                <div className={`step-indicator ${st}`}>
                  {st === 'pending'
                    ? <span style={{ fontSize: 12, color: 'var(--text3)' }}>{i + 1}</span>
                    : <StepIcon status={st} />
                  }
                </div>

                <div className="step-body" style={{ flex: 1 }}>
                  <div className="step-label-row">
                    <span className="step-label">{step.label}</span>
                    {st === 'running' && (
                      <span className="step-hint">{step.hint}</span>
                    )}
                    {canExpand && (
                      <button className="step-expand-btn" onClick={() => toggle(step.key)}>
                        {isOpen ? '▲' : '▼'}
                      </button>
                    )}
                  </div>

                  <div className="step-desc">
                    {st === 'running' && (
                      <span style={{ color: 'var(--accent2)' }}>
                        {step.runDesc}
                        <span className="msg-cursor" />
                      </span>
                    )}
                    {st === 'step-done' && (
                      <motion.span
                        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                        style={{ color: 'var(--green)' }}
                      >
                        {summary}
                      </motion.span>
                    )}
                    {st === 'error' && (
                      <span style={{ color: 'var(--red)' }}>{det?.message || 'Failed'}</span>
                    )}
                    {st === 'pending' && (
                      <span style={{ color: 'var(--text3)' }}>Waiting…</span>
                    )}
                  </div>

                  <AnimatePresence>
                    {isOpen && detail && (
                      <motion.div
                        className="step-detail"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.2 }}
                      >
                        {detail}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </motion.div>
            )
          })}
        </div>

        <AnimatePresence>
          {doneEvent && !rejected && !errorEvent && (
            <motion.div
              className="pipeline-done-banner"
              initial={{ opacity: 0, y: 10, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.4 }}
            >
              ✓ Analysis complete — opening chat
            </motion.div>
          )}
          {(rejected || errorEvent) && (
            <motion.div
              className="pipeline-rejected-banner"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              {rejected
                ? `Out of scope — ${doneEvent.scope_reason || 'no novel ML architecture found.'}`
                : errorEvent?.message || 'Pipeline failed unexpectedly.'}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  )
}
