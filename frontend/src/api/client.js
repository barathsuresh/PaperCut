const BASE = '/api'

async function json(res) {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try { detail = (await res.json()).detail || detail } catch {}
    throw new Error(detail)
  }
  return res.json()
}

export const getSessions = () => fetch(`${BASE}/sessions`).then(json)
export const getSession  = id => fetch(`${BASE}/sessions/${id}`).then(json)
export const getHistory  = id => fetch(`${BASE}/sessions/${id}/history`).then(json)
export const getArtifacts = id => fetch(`${BASE}/sessions/${id}/artifacts`).then(json)
export const deleteSession = id => fetch(`${BASE}/sessions/${id}`, { method: 'DELETE' }).then(json)
export const getArtifactContent = (id, group, file) =>
  fetch(`${BASE}/sessions/${id}/artifacts/${group}/${encodeURIComponent(file)}`).then(json)

export async function uploadPdf(formData) {
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: formData })
  return json(res)
}

export function streamPipeline(sessionId, onEvent) {
  const ctrl = new AbortController()
  ;(async () => {
    try {
      const res = await fetch(`${BASE}/run/pipeline/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
        signal: ctrl.signal,
      })
      const reader = res.body.getReader()
      const dec = new TextDecoder()
      let buf = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop()
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try { onEvent(JSON.parse(line.slice(6))) } catch {}
          }
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') onEvent({ type: 'error', message: e.message })
    }
  })()
  return () => ctrl.abort()
}

export function streamChat(sessionId, message, onChunk) {
  const ctrl = new AbortController()
  ;(async () => {
    try {
      const res = await fetch(`${BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message }),
        signal: ctrl.signal,
      })
      const reader = res.body.getReader()
      const dec = new TextDecoder()
      let buf = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop()
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try { onChunk(JSON.parse(line.slice(6))) } catch {}
          }
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') onChunk({ type: 'error', message: e.message })
    }
  })()
  return () => ctrl.abort()
}
