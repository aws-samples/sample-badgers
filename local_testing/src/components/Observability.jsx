import { useState, useEffect, useRef } from 'react'

export default function Observability() {
  const [sessions, setSessions] = useState([])
  const [sessionId, setSessionId] = useState('')
  const [hoursBack, setHoursBack] = useState(168)
  const [output, setOutput] = useState('')
  const [loading, setLoading] = useState(false)
  const abortRef = useRef(null)

  const refreshSessions = async () => {
    try {
      const r = await fetch('/api/chat-sessions')
      setSessions(await r.json())
    } catch { setSessions([]) }
  }

  useEffect(() => { refreshSessions() }, [])
  useEffect(() => { return () => { if (abortRef.current) abortRef.current.abort() } }, [])

  const fetchTraces = async (sid) => {
    const id = (sid || sessionId).trim()
    if (!id) return
    if (abortRef.current) abortRef.current.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setLoading(true)
    setOutput('Fetching traces...')
    try {
      const res = await fetch('/api/observability', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: id, hours_back: hoursBack }),
        signal: controller.signal,
      })
      const data = await res.json()
      setOutput(data.output || data.error || 'No traces found')
    } catch (e) {
      if (e.name === 'AbortError') return
      setOutput(`Error: ${e.message}`)
    }
    setLoading(false)
  }

  const selectSession = (sid) => {
    setSessionId(sid)
    if (sid) fetchTraces(sid)
  }

  return (
    <div>
      <div style={{ marginBottom: 12, fontSize: 13, color: 'var(--text-dim)' }}>
        View all traces and spans for a session from CloudWatch aws/spans
      </div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
        <select value={sessionId} onChange={e => selectSession(e.target.value)} style={{ flex: 1 }}>
          <option value="">Select a session...</option>
          {sessions.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <button onClick={refreshSessions} style={{ fontSize: 12 }}>↻ Refresh</button>
      </div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <input
          value={sessionId}
          onChange={e => setSessionId(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && fetchTraces()}
          placeholder="Or paste a session ID..."
          style={{ flex: 1, fontSize: 12 }}
        />
        <select value={hoursBack} onChange={e => setHoursBack(+e.target.value)} style={{ width: 130, fontSize: 12 }}>
          <option value={1}>Last 1 hour</option>
          <option value={6}>Last 6 hours</option>
          <option value={24}>Last 24 hours</option>
          <option value={72}>Last 3 days</option>
          <option value={168}>Last 7 days</option>
          <option value={720}>Last 30 days</option>
        </select>
        <button className="primary" onClick={() => fetchTraces()} disabled={loading || !sessionId.trim()}>
          {loading ? 'Fetching...' : '🔍 Fetch'}
        </button>
      </div>
      <div className="card" style={{ padding: 12 }}>
        <pre style={{
          fontSize: 12, fontFamily: 'SF Mono, Menlo, monospace',
          whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          minHeight: 400, maxHeight: 700, overflow: 'auto',
          color: output ? 'var(--text)' : 'var(--text-dim)',
        }}>
          {output || 'Select a session or paste an ID to view traces'}
        </pre>
      </div>
    </div>
  )
}
