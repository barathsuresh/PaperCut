import { useState, useCallback } from 'react'

let _id = 0

export default function useToast() {
  const [toasts, setToasts] = useState([])

  const toast = useCallback((message, type = 'info', duration = 3000) => {
    const id = ++_id
    setToasts(prev => [...prev, { id, message, type, duration }])
  }, [])

  const remove = useCallback(id => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  return { toasts, toast, remove }
}
