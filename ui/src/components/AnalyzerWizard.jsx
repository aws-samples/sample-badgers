import { useState } from 'react'

const MODELS = ['Claude Sonnet 4.5', 'Claude Haiku 4.5', 'Amazon Nova Premier', 'Claude Opus 4.6', 'Claude Opus 4.5']

export default function AnalyzerWizard({ runSSE, running }) {
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

  const set = (k, v) => setForm(p => ({ ...p, [k]: v }))

  const generatePrompts = async () => {
    setGenerating(true)
    setStatus('Generating prompts...')
    try {
      const res = await fetch('/api/wizard/generate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      const data = await res.json()
      if (data.error) { setStatus(`❌ ${data.error}`); setGenerating(false); return }
      setPrompts(data.prompts || {})
      setStatus('✓ Prompts generated')
      setStep(1)
    } catch (e) { setStatus(`❌ ${e.message}`) }
    setGenerating(false)
  }

  const generatePreview = async () => {
    try {
      const res = await fetch('/api/wizard/preview', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, prompts }),
      })
      setPreview(await res.json())
      setStep(3)
    } catch {}
  }

  const deploy = async () => {
    setDeployOutput('Saving analyzer...')
    try {
      const res = await fetch('/api/wizard/deploy', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, prompts, examples }),
      })
      const data = await res.json()
      setDeployOutput(data.output || data.error || 'Done')
    } catch (e) { setDeployOutput(`Error: ${e.message}`) }
  }

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

      {status && <div style={{ fontSize: 12, marginBottom: 12, color: status.startsWith('❌') ? 'var(--red)' : 'var(--green)' }}>{status}</div>}

      {step === 0 && (
        <div className="card" style={{ padding: 16 }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div>
              <label style={{ fontSize: 12, color: 'var(--text-dim)', display: 'block', marginBottom: 4 }}>Analyzer Name</label>
              <input value={form.displayName} onChange={e => set('displayName', e.target.value)} placeholder="e.g., Medical Form, Invoice, Blueprint" />
            </div>
            <div>
              <label style={{ fontSize: 12, color: 'var(--text-dim)', display: 'block', marginBottom: 4 }}>Short Description</label>
              <input value={form.description} onChange={e => set('description', e.target.value)} placeholder="One-line description" />
            </div>
            <div>
              <label style={{ fontSize: 12, color: 'var(--text-dim)', display: 'block', marginBottom: 4 }}>Detailed Description</label>
              <textarea value={form.details} onChange={e => set('details', e.target.value)} rows={5} placeholder="What should this analyzer look for?" />
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
            <button className="primary" onClick={deploy}>💾 Save Analyzer</button>
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
