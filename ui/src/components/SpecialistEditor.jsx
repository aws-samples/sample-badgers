import { useState, useEffect, useRef } from 'react'

const SPECIALIST_CATALOG = {
  'Content Extraction': {
    full_text_specialist: 'Full Text',
    elements_specialist: 'Elements',
    robust_elements_specialist: 'Robust Elements',
    table_specialist: 'Tables',
    handwriting_specialist: 'Handwriting',
    code_block_specialist: 'Code Blocks',
  },
  'Visual Analysis': {
    charts_specialist: 'Charts & Graphs',
    diagram_specialist: 'Diagrams',
    general_visual_analysis_specialist: 'General Visual',
    war_map_specialist: 'War Maps',
    decision_tree_specialist: 'Decision Trees',
  },
  'Document Intelligence': {
    page_specialist: 'Page Analysis',
    layout_specialist: 'Layout',
    keyword_topic_specialist: 'Keywords & Topics',
    correlation_specialist: 'Correlations',
    scientific_specialist: 'Scientific',
    editorial_specialist: 'Editorial',
    edu_transcript_specialist: 'Education Transcripts',
  },
  'Metadata': {
    metadata_generic_specialist: 'Generic Metadata',
    metadata_mods_specialist: 'MODS Metadata',
    metadata_mads_specialist: 'MADS Metadata',
    classify_pdf_content: 'PDF Classification',
  },
  'Processing': {
    image_enhancer: 'Image Enhancer',
    pdf_processor: 'PDF Processor',
    remediation_specialist: 'Remediation',
  },
}

function categorizeSpecialists(names) {
  const grouped = {}
  const uncategorized = []
  for (const name of names) {
    let found = false
    for (const [cat, map] of Object.entries(SPECIALIST_CATALOG)) {
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

export default function SpecialistEditor({ dirtyRef }) {
  const [specialists, setSpecialists] = useState([])
  const [selected, setSelected] = useState(null)
  const [prompts, setPrompts] = useState({})
  const [editing, setEditing] = useState({})
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const loadAbortRef = useRef(null)

  useEffect(() => {
    fetch('/api/specialists').then(r => r.json()).then(setSpecialists).catch(() => {})
  }, [])

  const isDirty = selected && JSON.stringify(editing) !== JSON.stringify(prompts)

  useEffect(() => {
    if (dirtyRef) dirtyRef.current = isDirty
    return () => { if (dirtyRef) dirtyRef.current = false }
  }, [isDirty, dirtyRef])

  const loadSpecialist = async (name) => {
    if (isDirty && !confirm('Unsaved changes. Discard?')) return
    if (loadAbortRef.current) loadAbortRef.current.abort()
    const controller = new AbortController()
    loadAbortRef.current = controller

    setSelected(name)
    setSaved(false)
    try {
      const res = await fetch(`/api/specialists/${name}/prompts`, { signal: controller.signal })
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
      await fetch(`/api/specialists/${selected}/prompts`, {
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
        <select value={selected || ''} onChange={e => e.target.value && loadSpecialist(e.target.value)} style={{ flex: 1 }}>
          <option value="">Select an specialist...</option>
          {Object.entries(categorizeSpecialists(specialists)).map(([cat, items]) => (
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
          {selected ? 'No prompts found for this specialist' : 'Select an specialist to edit its prompts'}
        </div>
      )}
    </div>
  )
}
