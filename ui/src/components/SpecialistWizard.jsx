import { useState } from 'react'

const MODELS = ['Claude Sonnet 4.5', 'Claude Haiku 4.5', 'Amazon Nova Premier', 'Claude Opus 4.6', 'Claude Opus 4.5']

export default function SpecialistWizard({ runSSE, running }) {
  const [step, setStep] = useState(0)
  const [form, setForm] = useState({
    displayName: '', description: '', details: '',
    primaryModel: MODELS[0], fallback1: MODELS[1], fallback2: MODELS[2],
    enhancement: false,
  })
  const [prompts, setPrompts] = useState({})
  const [generating, setGenerating] = useState(false)
  const [status, setStatus] = useState('')
  const [examples, setExamples] = useState([])
  const [preview, setPreview] = useState(null)
  const [deployOutput, setDeployOutput] = useState('')
  const [saved, setSaved] = useState(false)
  const [progress, setProgress] = useState(null)
  const [sectionLog, setSectionLog] = useState([])
  const [sections, setSections] = useState([])

  const set = (k, v) => setForm(p => ({ ...p, [k]: v }))

  // Generation streams over SSE: each section reports before it starts so the
  // status line can name what is being worked on.
  const generatePrompts = async () => {
    setGenerating(true)
    setProgress(null)
    setSectionLog([])
    setSections([])
    setStatus('Generating prompts...')
    let finished = null
    let failure = null
    try {
      const res = await fetch('/api/wizard/generate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          let msg
          try { msg = JSON.parse(line.slice(6)) } catch { continue }
          if (msg.type === 'start') {
            setSections(msg.sections || [])
          } else if (msg.type === 'progress') {
            setProgress(msg)
            setStatus(`Generating ${msg.index}/${msg.total} — ${msg.label}: ${msg.purpose}`)
          } else if (msg.type === 'section') {
            setSectionLog(prev => [...prev, msg])
          } else if (msg.type === 'done') {
            finished = msg
          } else if (msg.type === 'error') {
            failure = msg.message
          }
        }
      }
    } catch (e) { failure = e.message }

    setProgress(null)
    if (failure) { setStatus(`❌ ${failure}`); setGenerating(false); return }

    const generated = finished?.prompts || {}
    // Don't advance on an empty result — an empty step 2 looks like a skipped
    // step rather than a failure.
    if (!Object.keys(generated).length) {
      setStatus('❌ No prompts were returned')
      setGenerating(false)
      return
    }
    setPrompts(generated)
    const warnings = finished?.warnings || []
    setStatus(warnings.length
      ? `⚠ Generated with ${warnings.length} problem(s): ${warnings.join('; ')}`
      : `✓ Generated ${Object.keys(generated).length} prompts`)
    setStep(1)
    setGenerating(false)
  }

  const generatePreview = async () => {
    try {
      const res = await fetch('/api/wizard/preview', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, prompts, exampleCount: examples.length }),
      })
      const data = await res.json()
      if (data.error) { setStatus(`❌ ${data.error}`); return }
      setPreview(data)
      setStep(3)
    } catch (e) { setStatus(`❌ ${e.message}`) }
  }

  // Files are read here rather than sent as multipart so the save endpoint stays
  // a plain JSON handler.
  const readExamples = () => Promise.all(
    examples.map(file => new Promise((res, rej) => {
      const reader = new FileReader()
      reader.onload = () => res({ name: file.name, data: reader.result })
      reader.onerror = () => rej(new Error(`Could not read ${file.name}`))
      reader.readAsDataURL(file)
    }))
  )

  const save = async () => {
    setSaved(false)
    setDeployOutput('Saving specialist...')
    try {
      const encoded = await readExamples()
      const res = await fetch('/api/wizard/save', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, prompts, examples: encoded }),
      })
      const data = await res.json()
      setDeployOutput(data.output || data.error || 'Done')
      if (!data.error) setSaved(true)
    } catch (e) { setDeployOutput(`Error: ${e.message}`) }
  }

  // CDK runs for minutes, so this streams into the shared log panel.
  const deploy = () => runSSE('/api/wizard/deploy', {})

  const steps = ['Basic Info', 'Review Prompts', 'Examples', 'Deploy']

  return (
    <div>
      <div style={{ display: 'flex', gap: 4, marginBottom: 16 }}>
        {steps.map((s, i) => (
          <div key={i} style={{
            flex: 1, padding: '8px 12px', textAlign: 'center', fontSize: 12,
            background: i === step ? 'var(--accent-subtle)' : 'var(--surface)',
            border: `1px solid ${i === step ? 'var(--accent)' : 'var(--border)'}`,
            borderRadius: 'var(--radius)', color: i === step ? 'var(--accent)' : 'var(--text-dim)',
            cursor: i < step ? 'pointer' : 'default',
          }} onClick={() => i < step && setStep(i)}>
            {i + 1}. {s}
          </div>
        ))}
      </div>

      {/* Hidden while generating: the progress panel below already says the same
          thing, and two copies of the same line read as a glitch. */}
      {status && !generating && <div style={{ fontSize: 12, marginBottom: 12, color: status.startsWith('❌') ? 'var(--red)' : status.startsWith('⚠') ? 'var(--yellow, #b58900)' : 'var(--green)' }}>{status}</div>}

      {(progress || sectionLog.length > 0) && generating && (
        <div className="card" style={{ padding: 12, marginBottom: 12 }}>
          {progress && (
            <>
              <div style={{ fontSize: 12, marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                <span className="activity-spinner" aria-hidden="true" />
                <span style={{ color: 'var(--accent)' }}>Generating {progress.index}/{progress.total}</span>
                <span>—</span><strong>{progress.label}</strong>
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 8 }}>{progress.purpose}</div>
              {/* One segment per prompt section, so the checkpoints are the bar
                  rather than marks laid over it. Filled = done, half-lit = in
                  flight, red = failed, empty = pending. */}
              <div style={{ display: 'flex', gap: 3, marginBottom: 10 }}>
                {Array.from({ length: progress.total }, (_, i) => {
                  const done = sectionLog[i]
                  const current = i === progress.index - 1
                  const background = done
                    ? (done.ok ? 'var(--accent)' : 'var(--red)')
                    : current ? 'var(--accent)' : 'var(--surface)'
                  return (
                    <div
                      key={i}
                      title={sections[i]?.label || `Prompt ${i + 1}`}
                      style={{
                        flex: 1, height: 6, borderRadius: 2, background,
                        opacity: !done && current ? 0.4 : 1,
                        border: '1px solid var(--border)',
                        transition: 'background 200ms ease, opacity 200ms ease',
                      }}
                    />
                  )
                })}
              </div>
            </>
          )}
          {sectionLog.map((s, i) => (
            <div key={i} style={{ fontSize: 11, fontFamily: 'SF Mono, Menlo, monospace', color: s.ok ? 'var(--text-dim)' : 'var(--red)' }}>
              {s.ok ? `✓ ${s.label} (${s.chars.toLocaleString()} chars)` : `✗ ${s.label} — ${s.message}`}
            </div>
          ))}
        </div>
      )}

      {step === 0 && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div>
              <label style={{ fontSize: 12, color: 'var(--text-dim)', display: 'block', marginBottom: 4 }}>Specialist Name</label>
              <input value={form.displayName} onChange={e => set('displayName', e.target.value)} placeholder="e.g., Medical Form, Invoice, Blueprint" />
            </div>
            <div>
              <label style={{ fontSize: 12, color: 'var(--text-dim)', display: 'block', marginBottom: 4 }}>Short Description</label>
              <input value={form.description} onChange={e => set('description', e.target.value)} placeholder="One-line description" />
            </div>
            <div>
              <label style={{ fontSize: 12, color: 'var(--text-dim)', display: 'block', marginBottom: 4 }}>Detailed Description</label>
              <textarea value={form.details} onChange={e => set('details', e.target.value)} rows={5} placeholder="What should this specialist look for?" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
              {[['primaryModel', 'Primary Model'], ['fallback1', 'Fallback 1'], ['fallback2', 'Fallback 2']].map(([k, label]) => (
                <div key={k}>
                  <label style={{ fontSize: 12, color: 'var(--text-dim)', display: 'block', marginBottom: 4 }}>{label}</label>
                  <select value={form[k]} onChange={e => set(k, e.target.value)}>
                    {MODELS.map(m => <option key={m} value={m}>{m}</option>)}
                  </select>
                </div>
              ))}
            </div>
            <label style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
              <input type="checkbox" checked={form.enhancement} onChange={e => set('enhancement', e.target.checked)} />
              Enhancement eligible? (for busy/degraded/historical documents)
            </label>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <button className="primary" onClick={generatePrompts} disabled={generating || !form.displayName || !form.description}>
                {generating ? 'Generating...' : 'Generate Prompts →'}
              </button>
            </div>
          </div>
        </div>
      )}

      {step === 1 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {Object.entries(prompts).map(([key, val]) => (
            <div key={key} className="card" style={{ padding: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 6, color: 'var(--accent)' }}>{key}</div>
              <textarea value={val} onChange={e => setPrompts(p => ({ ...p, [key]: e.target.value }))} rows={8} style={{ width: '100%', fontSize: 12 }} />
            </div>
          ))}
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <button onClick={() => setStep(0)}>← Back</button>
            <button className="primary" onClick={() => setStep(2)}>Continue →</button>
          </div>
        </div>
      )}

      {step === 2 && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ fontSize: 13, marginBottom: 12 }}>Upload example images (optional, max 6)</div>
          <input type="file" multiple accept="image/*" onChange={e => setExamples([...e.target.files].slice(0, 6))} />
          {examples.length > 0 && <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 8 }}>{examples.length} file(s) selected</div>}
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 16 }}>
            <button onClick={() => setStep(1)}>← Back</button>
            <button className="primary" onClick={generatePreview}>Preview Config →</button>
          </div>
        </div>
      )}

      {step === 3 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {preview && (
            <div className="card" style={{ padding: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 8 }}>Configuration Preview</div>
              <pre style={{ fontSize: 11, fontFamily: 'SF Mono, Menlo, monospace', whiteSpace: 'pre-wrap', maxHeight: 300, overflow: 'auto' }}>
                {JSON.stringify(preview, null, 2)}
              </pre>
            </div>
          )}
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <button onClick={() => setStep(2)}>← Back</button>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="primary" onClick={save} disabled={running}>💾 Save Specialist</button>
              <button onClick={deploy} disabled={!saved || running} title={saved ? 'Deploy the custom specialists stack' : 'Save first'}>
                ☁️ Deploy Stack
              </button>
            </div>
          </div>
          {deployOutput && (
            <div className="card" style={{ padding: 12 }}>
              <pre style={{ fontSize: 12, fontFamily: 'SF Mono, Menlo, monospace', whiteSpace: 'pre-wrap' }}>{deployOutput}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
