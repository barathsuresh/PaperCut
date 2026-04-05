import { useEffect } from 'react'
import { AnimatePresence, motion } from 'framer-motion'

export default function ToastContainer({ toasts, onRemove }) {
  return (
    <div className="toast-container">
      <AnimatePresence>
        {toasts.map(t => (
          <Toast key={t.id} toast={t} onRemove={onRemove} />
        ))}
      </AnimatePresence>
    </div>
  )
}

function Toast({ toast, onRemove }) {
  useEffect(() => {
    const timer = setTimeout(() => onRemove(toast.id), toast.duration ?? 3000)
    return () => clearTimeout(timer)
  }, [toast.id, toast.duration, onRemove])

  return (
    <motion.div
      className={`toast toast-${toast.type ?? 'info'}`}
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 10, scale: 0.95 }}
      transition={{ duration: 0.2 }}
      onClick={() => onRemove(toast.id)}
    >
      <span className="toast-icon">
        {toast.type === 'success' ? '✓' : toast.type === 'error' ? '✕' : 'ℹ'}
      </span>
      {toast.message}
    </motion.div>
  )
}
