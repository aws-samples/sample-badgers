import { useState } from 'react'

export default function Observability() {
  const [sessionId, setSessionId] = useState('')
  const [output, setOutput] = useState('')
  const [loading, setLoading] = useState(false)

  const fetchTraces = async () => {
    if (!sessionId.trim()) return
    setLoading(true)
    setOutput('Fetching traces...')
    try {
      const res = await fetch('/api/observability', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId.trim() }),
      })
      const data = await res.json()
      setOutput(data.output || data.error || 'No traces found')
    } catch (e) { setOutput(`Error: ${e.message}`) }
    setLoading(false)
  }

  return (
    <div>
      <div style={{ marginBottom: 12, fontSize: 13, color: 'var(--text-dim)' }}>
        View all traces and spans for a session from CloudWatch aws/spans (last 24 hours)
      </div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <input
          value={sessionId}
          onChange={e => setSessionId(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && fetchTraces()}
          placeholder="ws-session-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
          style={{ flex: 1 }}
        />
        <button className="primary" onClick={fetchTraces} disabled={loading || !sessionId.trim()}>
          {loading ? 'Fetching...' : '🔍 Fetch Session'}
        </button>
      </div>
      <div className="card" style={{ padding: 12 }}>
        <pre style={{
          fontSize: 12, fontFamily: 'SF Mono, Menlo, monospace',
          whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          minHeight: 400, maxHeight: 700, overflow: 'auto',
          color: output ? 'var(--text)' : 'var(--text-dim)',
        }}>
          {output || 'Enter a session ID and click Fetch to view traces'}
        </pre>
      </div>
    </div>
  )
}
