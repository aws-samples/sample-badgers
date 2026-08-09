import React, { useEffect, useMemo, useState } from 'react'
import JsonHighlighter from './JsonHighlighter.jsx'

export default function RecognitionOutput() {
  const [sessions, setSessions] = useState([])
  const [selected, setSelected] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [copyStatus, setCopyStatus] = useState('')

  const formatted = useMemo(() => result ? JSON.stringify(result, null, 2) : '', [result])

  async function loadSession(sessionId) {
    if (!sessionId) return
    setSelected(sessionId)
    setLoading(true)
    setError('')
    setCopyStatus('')
    try {
      const response = await fetch(`/api/recognition/sessions/${encodeURIComponent(sessionId)}`)
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Unable to load recognition output')
      setResult(data)
    } catch (loadError) {
      setResult(null)
      setError(loadError.message)
    } finally {
      setLoading(false)
    }
  }

  async function refresh() {
    setLoading(true)
    setError('')
    try {
      const response = await fetch('/api/recognition/sessions')
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Unable to list recognition sessions')
      setSessions(data)
      const nextSession = selected || data[0]?.sessionId
      if (nextSession) await loadSession(nextSession)
      else {
        setResult(null)
        setLoading(false)
      }
    } catch (refreshError) {
      setSessions([])
      setResult(null)
      setError(refreshError.message)
      setLoading(false)
    }
  }

  useEffect(() => { refresh() }, [])

  async function copyJson() {
    try {
      await navigator.clipboard.writeText(formatted)
      setCopyStatus('Copied')
    } catch {
      setCopyStatus('Copy failed')
    }
  }

  function downloadJson() {
    const blob = new Blob([formatted], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${selected || 'recognition-output'}.json`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <select
          aria-label="Recognition session"
          value={selected}
          onChange={event => loadSession(event.target.value)}
          style={{ flex: '1 1 420px' }}
        >
          <option value="">Select a recognition session...</option>
          {sessions.map(session => (
            <option key={session.sessionId} value={session.sessionId}>
              {session.sessionId} — {session.recognitionFileCount} JSON file{session.recognitionFileCount === 1 ? '' : 's'}
            </option>
          ))}
        </select>
        <button onClick={refresh} disabled={loading}>↻ Refresh</button>
        <button onClick={copyJson} disabled={!formatted}>Copy JSON</button>
        <button className="primary" onClick={downloadJson} disabled={!formatted}>Download JSON</button>
        {copyStatus && <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>{copyStatus}</span>}
      </div>

      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '10px 12px', borderBottom: '1px solid var(--border)', fontSize: 13, fontWeight: 600 }}>
          Recognition JSON
        </div>
        {loading ? (
          <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-dim)' }}>Loading recognition output…</div>
        ) : error ? (
          <div style={{ padding: 20, color: 'var(--red)' }}>{error}</div>
        ) : formatted ? (
          <div style={{ minHeight: 500, maxHeight: '72vh', overflow: 'auto' }}>
            <JsonHighlighter text={formatted} />
          </div>
        ) : (
          <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-dim)' }}>No recognition JSON is available.</div>
        )}
      </div>
    </div>
  )
}
