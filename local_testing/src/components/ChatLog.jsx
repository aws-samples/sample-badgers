import { useState, useEffect } from 'react'

export default function ChatLog() {
  const [sessions, setSessions] = useState([])
  const [selected, setSelected] = useState(null)
  const [content, setContent] = useState('')

  const refresh = async () => {
    try {
      const res = await fetch('/api/chat-sessions')
      setSessions(await res.json())
    } catch { setSessions([]) }
  }

  useEffect(() => { refresh() }, [])

  const loadSession = async (sid) => {
    setSelected(sid)
    try {
      const res = await fetch(`/api/chat-sessions/${sid}`)
      const data = await res.json()
      setContent(data.content || 'Empty session')
    } catch (e) { setContent(`Error: ${e.message}`) }
  }

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'center' }}>
        <select
          value={selected || ''}
          onChange={e => e.target.value && loadSession(e.target.value)}
          style={{ flex: 1 }}
        >
          <option value="">Select a session...</option>
          {sessions.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <button onClick={refresh} style={{ fontSize: 12 }}>↻ Refresh</button>
      </div>
      <div className="card" style={{ padding: 12 }}>
        <pre style={{
          fontSize: 12, fontFamily: 'SF Mono, Menlo, monospace',
          whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          minHeight: 400, maxHeight: 700, overflow: 'auto',
          color: content ? 'var(--text)' : 'var(--text-dim)',
        }}>
          {content || 'Select a session to view its chat history'}
        </pre>
      </div>
    </div>
  )
}
