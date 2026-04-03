import { useState, useEffect, useRef } from 'react'

const ANALYZER_CATALOG = {
  'Content Extraction': {
    full_text_analyzer: 'Full Text',
    elements_analyzer: 'Elements',
    robust_elements_analyzer: 'Robust Elements',
    table_analyzer: 'Tables',
    handwriting_analyzer: 'Handwriting',
    code_block_analyzer: 'Code Blocks',
  },
  'Visual Analysis': {
    charts_analyzer: 'Charts & Graphs',
    diagram_analyzer: 'Diagrams',
    general_visual_analysis_analyzer: 'General Visual',
    war_map_analyzer: 'War Maps',
    decision_tree_analyzer: 'Decision Trees',
  },
  'Document Intelligence': {
    page_analyzer: 'Page Analysis',
    layout_analyzer: 'Layout',
    keyword_topic_analyzer: 'Keywords & Topics',
    correlation_analyzer: 'Correlations',
    scientific_analyzer: 'Scientific',
    editorial_analyzer: 'Editorial',
    edu_transcript_analyzer: 'Education Transcripts',
  },
  'Metadata': {
    metadata_generic_analyzer: 'Generic Metadata',
    metadata_mods_analyzer: 'MODS Metadata',
    metadata_mads_analyzer: 'MADS Metadata',
    classify_pdf_content: 'PDF Classification',
  },
  'Processing': {
    image_enhancer: 'Image Enhancer',
    pdf_processor: 'PDF Processor',
    remediation_analyzer: 'Remediation',
  },
}

function categorizeAnalyzers(names) {
  const grouped = {}
  const uncategorized = []
  for (const name of names) {
    let found = false
    for (const [cat, map] of Object.entries(ANALYZER_CATALOG)) {
      if (map[name]) {
        if (!grouped[cat]) grouped[cat] = []
        grouped[cat].push({ key: name, label: map[name] })
        found = true
        break
      }
    }
    if (!found) uncategorized.push({ key: name, label: name })
  }
  if (uncategorized.length) grouped['Other'] = uncategorized
  return grouped
}

export default function AnalyzerEditor({ dirtyRef }) {
  const [analyzers, setAnalyzers] = useState([])
  const [selected, setSelected] = useState(null)
  const [prompts, setPrompts] = useState({})
  const [editing, setEditing] = useState({})
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const loadAbortRef = useRef(null)

  useEffect(() => {
    fetch('/api/analyzers').then(r => r.json()).then(setAnalyzers).catch(() => {})
  }, [])

  const isDirty = selected && JSON.stringify(editing) !== JSON.stringify(prompts)

  useEffect(() => {
    if (dirtyRef) dirtyRef.current = isDirty
    return () => { if (dirtyRef) dirtyRef.current = false }
  }, [isDirty, dirtyRef])

  const loadAnalyzer = async (name) => {
    if (isDirty && !confirm('Unsaved changes. Discard?')) return
    if (loadAbortRef.current) loadAbortRef.current.abort()
    const controller = new AbortController()
    loadAbortRef.current = controller

    setSelected(name)
    setSaved(false)
    try {
      const res = await fetch(`/api/analyzers/${name}/prompts`, { signal: controller.signal })
      const data = await res.json()
      setPrompts(data)
      setEditing(data)
    } catch (e) {
      if (e.name === 'AbortError') return
      setPrompts({}); setEditing({})
    }
  }

  const save = async () => {
    setSaving(true)
    try {
      await fetch(`/api/analyzers/${selected}/prompts`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editing),
      })
      setPrompts(editing)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch {}
    setSaving(false)
  }

  const promptKeys = Object.keys(editing).sort()

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'center' }}>
        <select value={selected || ''} onChange={e => e.target.value && loadAnalyzer(e.target.value)} style={{ flex: 1 }}>
          <option value="">Select an analyzer...</option>
          {Object.entries(categorizeAnalyzers(analyzers)).map(([cat, items]) => (
            <optgroup key={cat} label={cat}>
              {items.map(a => <option key={a.key} value={a.key}>{a.label}</option>)}
            </optgroup>
          ))}
        </select>
        {isDirty && <span style={{ fontSize: 11, color: 'var(--yellow)' }}>Unsaved</span>}
        {saved && <span style={{ fontSize: 11, color: 'var(--green)' }}>✓ Saved</span>}
        <button className="primary" onClick={save} disabled={saving || !isDirty}>
          {saving ? 'Saving...' : 'Save'}
        </button>
      </div>

      {selected && promptKeys.length > 0 ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {promptKeys.map(key => (
            <div key={key} className="card" style={{ padding: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 6, color: 'var(--accent)' }}>{key}</div>
              <textarea
                value={editing[key] || ''}
                onChange={e => setEditing(prev => ({ ...prev, [key]: e.target.value }))}
                rows={10}
                style={{ width: '100%', fontSize: 12 }}
              />
            </div>
          ))}
        </div>
      ) : (
        <div className="card" style={{ padding: 40, textAlign: 'center', color: 'var(--text-dim)', fontSize: 13 }}>
          {selected ? 'No prompts found for this analyzer' : 'Select an analyzer to edit its prompts'}
        </div>
      )}
    </div>
  )
}
