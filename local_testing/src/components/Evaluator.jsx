import { useState, useEffect } from 'react'

export default function Evaluator() {
  const [sessions, setSessions] = useState([])
  const [selected, setSelected] = useState(null)
  const [results, setResults] = useState([])
  const [ratings, setRatings] = useState({})
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    fetch('/api/eval/sessions').then(r => r.json()).then(setSessions).catch(() => {})
  }, [])

  const loadSession = async (sid) => {
    setSelected(sid)
    try {
      const res = await fetch(`/api/eval/sessions/${sid}`)
      const data = await res.json()
      setResults(data.results || [])
      setRatings(data.ratings || {})
    } catch { setResults([]); setRatings({}) }
  }

  const setRating = (resultId, questionId, value) => {
    setRatings(prev => ({
      ...prev,
      [resultId]: { ...(prev[resultId] || {}), [questionId]: value }
    }))
  }

  const save = async () => {
    setSaving(true)
    try {
      await fetch(`/api/eval/sessions/${selected}/ratings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ratings),
      })
    } catch {}
    setSaving(false)
  }

  const LIKERT = ['Failed', 'Major errors', 'Partial success', 'Minor issues', 'Perfect']

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'center' }}>
        <select value={selected || ''} onChange={e => e.target.value && loadSession(e.target.value)} style={{ flex: 1 }}>
          <option value="">Select a session to evaluate...</option>
          {sessions.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <button className="primary" onClick={save} disabled={saving || !selected}>
          {saving ? 'Saving...' : 'Save Ratings'}
        </button>
      </div>

      {results.length > 0 ? results.map((r, i) => (
        <div key={i} className="card" style={{ padding: 12, marginBottom: 8 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 4 }}>{r.analyzer || `Result ${i + 1}`}</div>
          <pre style={{ fontSize: 11, fontFamily: 'SF Mono, Menlo, monospace', whiteSpace: 'pre-wrap', maxHeight: 200, overflow: 'auto', marginBottom: 12, color: 'var(--text-dim)' }}>
            {r.content?.substring(0, 500) || 'No content'}
            {r.content?.length > 500 ? '...' : ''}
          </pre>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {(r.questions || [{ id: 'overall', text: 'Overall accuracy' }]).map(q => (
              <div key={q.id} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 12, color: 'var(--text-dim)', minWidth: 200 }}>{q.text}</span>
                <div style={{ display: 'flex', gap: 2 }}>
                  {LIKERT.map((label, li) => (
                    <button key={li} onClick={() => setRating(r.id || i, q.id, li + 1)}
                      style={{
                        fontSize: 11, padding: '2px 8px',
                        background: (ratings[r.id || i]?.[q.id] === li + 1) ? 'var(--accent)' : 'var(--surface)',
                        borderColor: (ratings[r.id || i]?.[q.id] === li + 1) ? 'var(--accent)' : 'var(--border)',
                      }}>
                      {li + 1}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )) : (
        <div className="card" style={{ padding: 40, textAlign: 'center', color: 'var(--text-dim)', fontSize: 13 }}>
          {selected ? 'No results found for this session' : 'Select a session to evaluate results'}
        </div>
      )}
    </div>
  )
}
