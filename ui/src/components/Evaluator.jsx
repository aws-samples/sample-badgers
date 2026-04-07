import { useState, useEffect, useCallback } from 'react'

const LIKERT = ['Failed', 'Major errors', 'Partial success', 'Minor issues', 'Perfect']

function formatMetadata(meta) {
  if (!meta) return null
  const lines = []
  const f = meta.input_file || {}
  if (f.name) lines.push(`**File:** \`${f.name}\``)
  if (meta.timestamp_completed) {
    try {
      const d = new Date(meta.timestamp_completed)
      lines.push(`**Processed:** ${d.toISOString().slice(0, 19).replace('T', ' ')} UTC`)
    } catch { lines.push(`**Processed:** ${meta.timestamp_completed}`) }
  }
  const s = meta.stats || {}
  if (s.total_analyses_performed != null)
    lines.push(`**Analyses:** ${s.successful_analyses || 0}/${s.total_analyses_performed} successful across ${s.pages_with_content || 0} pages`)
  const cs = meta.content_summary || {}
  if (Object.keys(cs).length) {
    lines.push('', '**Content Detected:**')
    for (const [tool, info] of Object.entries(cs)) {
      const count = info.count || 0
      const pages = [...new Set(info.pages || [])].sort((a, b) => a - b)
      lines.push(`- \`${tool}\`: ${count} on pages [${pages.join(', ')}]`)
    }
  }
  return lines.join('\n')
}

export default function Evaluator() {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState(null)
  const [results, setResults] = useState([])
  const [metadata, setMetadata] = useState(null)
  const [allRatings, setAllRatings] = useState({})
  const [index, setIndex] = useState(0)
  const [content, setContent] = useState('')
  const [loadingContent, setLoadingContent] = useState(false)
  const [toolQuestions, setToolQuestions] = useState([])
  const [saving, setSaving] = useState(false)
  const [saveMsg, setSaveMsg] = useState('')

  const currentResult = results[index] || null
  const currentRating = currentResult ? (allRatings[currentResult.filename] || {}) : {}

  const refreshSessions = async () => {
    setLoading(true)
    try {
      const r = await fetch('/api/eval/sessions')
      setSessions(await r.json())
    } catch { setSessions([]) }
    setLoading(false)
  }

  useEffect(() => { refreshSessions() }, [])

  const loadSession = async (sid) => {
    setSelected(sid)
    setIndex(0)
    setContent('')
    setToolQuestions([])
    setSaveMsg('')
    try {
      const r = await fetch(`/api/eval/sessions/${sid}`)
      const data = await r.json()
      setResults(data.results || [])
      setMetadata(data.metadata || null)
      setAllRatings(data.ratings || {})
    } catch {
      setResults([])
      setMetadata(null)
      setAllRatings({})
    }
  }

  const loadResultContent = useCallback(async (result) => {
    if (!result) { setContent(''); setToolQuestions([]); return }
    setLoadingContent(true)
    try {
      const r = await fetch(`/api/eval/result?key=${encodeURIComponent(result.key)}`)
      const data = await r.json()
      setContent(data.content || '')
    } catch { setContent('Error loading content') }
    // Load tool-specific eval questions from manifest
    try {
      const r = await fetch(`/api/eval/manifest-eval/${result.analyzer}`)
      const data = await r.json()
      const qs = data.evaluation?.questions?.tool_specific || []
      setToolQuestions(qs)
    } catch { setToolQuestions([]) }
    setLoadingContent(false)
  }, [selected])

  useEffect(() => {
    if (results.length > 0 && results[index]) loadResultContent(results[index])
  }, [index, results, loadResultContent])

  const setField = (field, value) => {
    if (!currentResult) return
    setAllRatings(prev => ({
      ...prev,
      [currentResult.filename]: { ...(prev[currentResult.filename] || {}), [field]: value }
    }))
  }

  const autoSave = async () => {
    if (!selected || !currentResult) return
    const responses = allRatings[currentResult.filename] || {}
    if (responses.overall_accuracy == null && responses.element_identification == null) return
    try {
      await fetch(`/api/eval/sessions/${selected}/ratings`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ result_file: currentResult.filename, analyzer: currentResult.analyzer, responses }),
      })
    } catch {}
  }

  const navigate = async (dir) => {
    await autoSave()
    setSaveMsg('')
    setIndex(prev => Math.max(0, Math.min(results.length - 1, prev + dir)))
  }

  const save = async () => {
    if (!selected || !currentResult) return
    setSaving(true)
    setSaveMsg('')
    const responses = allRatings[currentResult.filename] || {}
    try {
      const r = await fetch(`/api/eval/sessions/${selected}/ratings`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ result_file: currentResult.filename, analyzer: currentResult.analyzer, responses }),
      })
      const data = await r.json()
      setSaveMsg(data.ok ? '✅ Saved' : `❌ ${data.error || 'Failed'}`)
    } catch (e) { setSaveMsg(`❌ ${e.message}`) }
    setSaving(false)
  }

  const metaText = formatMetadata(metadata)

  return (
    <div>
      {/* Session selector */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'center' }}>
        <select value={selected || ''} onChange={e => e.target.value && loadSession(e.target.value)} style={{ flex: 1 }}>
          <option value="">Select a session to evaluate...</option>
          {sessions.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <button onClick={refreshSessions} disabled={loading}>
          {loading ? 'Loading...' : '🔄 Refresh'}
        </button>
      </div>

      {/* Session metadata */}
      {metaText && (
        <div className="card" style={{ padding: 12, marginBottom: 12, fontSize: 12, lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
          {metaText}
        </div>
      )}

      {/* Navigation */}
      {results.length > 0 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
          <button onClick={() => navigate(-1)} disabled={index === 0}>◀ Previous</button>
          <span style={{ fontSize: 20, fontWeight: 700, flex: 1, textAlign: 'center' }}>
            {index + 1} / {results.length}
          </span>
          <button onClick={() => navigate(1)} disabled={index >= results.length - 1}>Next ▶</button>
        </div>
      )}

      {currentResult && (
        <div style={{ fontSize: 12, marginBottom: 8, color: 'var(--text-dim)' }}>
          <span style={{ marginRight: 16 }}>File: <code>{currentResult.filename}</code></span>
          <span>Analyzer: <code>{currentResult.analyzer}</code></span>
        </div>
      )}

      {/* Two-column: result + evaluation */}
      {results.length > 0 ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {/* Left: result content */}
          <div className="card" style={{ padding: 12 }}>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 8 }}>📄 Result Output</div>
            {loadingContent ? (
              <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-dim)' }}>Loading...</div>
            ) : (
              <pre style={{
                fontSize: 11, fontFamily: 'SF Mono, Menlo, monospace', whiteSpace: 'pre-wrap',
                maxHeight: 600, overflow: 'auto', color: 'var(--text-dim)', margin: 0,
              }}>
                {content || 'No content'}
              </pre>
            )}
          </div>

          {/* Right: evaluation form */}
          <div className="card" style={{ padding: 12 }}>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>✅ Evaluation</div>

            <LikertQuestion label="Overall accuracy" field="overall_accuracy" value={currentRating.overall_accuracy} onChange={setField} />
            <LikertQuestion label="Were all visual elements correctly identified?" field="element_identification" value={currentRating.element_identification} onChange={setField} />
            <LikertQuestion label="Was content understood within surrounding context?" field="contextual_understanding" value={currentRating.contextual_understanding} onChange={setField} />

            <div style={{ marginBottom: 12 }}>
              <label style={{ fontSize: 12, display: 'block', marginBottom: 4, color: 'var(--text-dim)' }}>
                What elements were missed or incorrectly represented?
              </label>
              <textarea rows={3} value={currentRating.issues_noted || ''} onChange={e => setField('issues_noted', e.target.value)}
                style={{ width: '100%', fontSize: 12, fontFamily: 'SF Mono, Menlo, monospace', resize: 'vertical' }} />
            </div>

            {/* Tool-specific questions */}
            {toolQuestions.length > 0 && (
              <>
                <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 8, marginTop: 8, borderTop: '1px solid var(--border)', paddingTop: 8 }}>
                  Tool-Specific Questions
                </div>
                {toolQuestions.map((q, qi) => (
                  <LikertQuestion key={qi} label={q.text} field={`tool_q${qi + 1}`} value={currentRating[`tool_q${qi + 1}`]} onChange={setField} />
                ))}
              </>
            )}

            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 8 }}>
              <button className="primary" onClick={save} disabled={saving}>
                {saving ? 'Saving...' : '💾 Save Evaluation'}
              </button>
              {saveMsg && <span style={{ fontSize: 12 }}>{saveMsg}</span>}
            </div>
          </div>
        </div>
      ) : (
        <div className="card" style={{ padding: 40, textAlign: 'center', color: 'var(--text-dim)', fontSize: 13 }}>
          {selected ? 'No results found for this session' : 'Click Refresh to load sessions from S3, then select one'}
        </div>
      )}
    </div>
  )
}

function LikertQuestion({ label, field, value, onChange }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>{label}</div>
      <div style={{ display: 'flex', gap: 2 }}>
        {LIKERT.map((lbl, i) => (
          <button key={i} onClick={() => onChange(field, i + 1)}
            title={lbl}
            style={{
              fontSize: 11, padding: '2px 8px',
              background: value === i + 1 ? 'var(--accent)' : 'var(--surface)',
              borderColor: value === i + 1 ? 'var(--accent)' : 'var(--border)',
              color: value === i + 1 ? '#fff' : 'inherit',
            }}>
            {i + 1}
          </button>
        ))}
        <span style={{ fontSize: 10, color: 'var(--text-dim)', marginLeft: 4, alignSelf: 'center' }}>
          {value ? LIKERT[value - 1] : ''}
        </span>
      </div>
    </div>
  )
}
