import { useState, useEffect, useMemo } from 'react'

const FIXED_IMAGE_TOKENS = 1600

const DEFAULT_INCLUDED = new Set([
  'charts_analyzer', 'classify_pdf_content', 'correlation_analyzer',
  'diagram_analyzer', 'elements_analyzer', 'general_visual_analysis',
  'handwriting_analyzer', 'keyword_topic_analyzer', 'pdf_processor',
  'robust_elements_analyzer', 'table_analyzer',
])

function fmt(n, decimals = 0) {
  return Number(n).toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
}

function ResultCard({ label, value }) {
  return (
    <div className="card" style={{ padding: 10, textAlign: 'center' }}>
      <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>{label}</div>
      <div style={{ fontSize: 15, fontWeight: 600, fontFamily: 'SF Mono, Menlo, monospace' }}>{value}</div>
    </div>
  )
}

function Field({ label, value, onChange, type = 'number', step, min, max, disabled }) {
  return (
    <div>
      <label style={{ fontSize: 11, color: 'var(--text-dim)', display: 'block', marginBottom: 2 }}>{label}</label>
      <input type={type} value={value} onChange={e => onChange(Number(e.target.value))}
        step={step} min={min} max={max} disabled={disabled} style={{ width: '100%' }} />
    </div>
  )
}

// ── Basic Calculator Tab ──
function BasicCalculator({ config }) {
  const presets = Object.values(config.presets || {})
  const models = Object.values(config.models || {})
  const modelNames = models.map(m => m.name)
  const ing = config.ingestion || {}

  const [preset, setPreset] = useState(presets[0]?.name || '')
  const [presetDesc, setPresetDesc] = useState('')
  const [model, setModel] = useState(modelNames[0] || '')
  const [cpt, setCpt] = useState(ing.characters_per_token || 4.5)
  const [cpw, setCpw] = useState(ing.avg_characters_per_word || 5)
  const [wpp, setWpp] = useState(ing.avg_words_per_page || 500)
  const [ppd, setPpd] = useState(ing.avg_pages_per_document || 15)
  const [tpi, setTpi] = useState(ing.avg_tokens_per_image || 1600)
  const [docs, setDocs] = useState(100)
  const [imgs, setImgs] = useState(50)
  const [ratio, setRatio] = useState(0.5)
  const [result, setResult] = useState(null)

  const modelInfo = models.find(m => m.name === model) || models[0] || {}
  const inpM = modelInfo.input_cost_per_million || 0
  const outM = modelInfo.output_cost_per_million || 0

  // Derived values
  const charsPerPage = wpp * cpw
  const tokensPerPage = charsPerPage / cpt
  const wordsPerDoc = wpp * ppd
  const charsPerDoc = charsPerPage * ppd
  const tokensPerDoc = charsPerDoc / cpt

  const applyPreset = (name) => {
    setPreset(name)
    const p = presets.find(x => x.name === name)
    if (!p) return
    setPresetDesc(`${p.name}: ${p.description}`)
    setModel(p.recommended_model)
    setWpp(p.words_per_page)
    setPpd(p.pages_per_document)
    setTpi(p.tokens_per_image)
    setRatio(p.output_ratio)
  }

  useEffect(() => { if (presets[0]) applyPreset(presets[0].name) }, [])

  const calculate = () => {
    const totalInput = (docs * tokensPerDoc) + (imgs * tpi)
    const totalOutput = totalInput * ratio
    const inputCost = (totalInput / 1e6) * inpM
    const outputCost = (totalOutput / 1e6) * outM
    const total = inputCost + outputCost
    const costPerDoc = docs > 0 ? total / docs : 0
    const costPerPage = (docs > 0 && ppd > 0) ? total / (docs * ppd) : 0
    setResult({ totalInput, totalOutput, inputCost, outputCost, total, costPerDoc, costPerPage })
  }

  const reset = () => {
    setCpt(ing.characters_per_token || 4.5)
    setCpw(ing.avg_characters_per_word || 5)
    setWpp(ing.avg_words_per_page || 500)
    setPpd(ing.avg_pages_per_document || 15)
    setTpi(ing.avg_tokens_per_image || 1600)
    setDocs(100); setImgs(50); setRatio(0.5); setResult(null)
  }

  return (
    <div>
      <div style={{ fontSize: 13, color: 'var(--text-dim)', marginBottom: 12 }}>
        Estimate Bedrock costs for document/image analysis. Select an industry preset or adjust manually.
      </div>

      {/* Preset */}
      <div className="card" style={{ padding: 12, marginBottom: 12 }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
          <span style={{ fontSize: 12, fontWeight: 500 }}>🏭 Industry Preset</span>
          <select value={preset} onChange={e => applyPreset(e.target.value)} style={{ flex: 1 }}>
            {presets.map(p => <option key={p.name} value={p.name}>{p.name}</option>)}
          </select>
        </div>
        {presetDesc && <div style={{ fontSize: 12, color: 'var(--text-dim)', fontStyle: 'italic' }}>{presetDesc}</div>}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
        {/* Model & Pricing */}
        <div className="card" style={{ padding: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 8 }}>🤖 Model Selection</div>
          <select value={model} onChange={e => setModel(e.target.value)} style={{ marginBottom: 8, width: '100%' }}>
            {modelNames.map(n => <option key={n} value={n}>{n}</option>)}
          </select>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
            <ResultCard label="Input $/M tokens" value={`$${inpM.toFixed(2)}`} />
            <ResultCard label="Output $/M tokens" value={`$${outM.toFixed(2)}`} />
            <ResultCard label="Input $/token" value={`$${(inpM / 1e6).toFixed(10)}`} />
            <ResultCard label="Output $/token" value={`$${(outM / 1e6).toFixed(10)}`} />
          </div>
        </div>

        {/* Ingestion Values */}
        <div className="card" style={{ padding: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 8 }}>📊 Ingestion Values</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
            <Field label="Chars/Token" value={cpt} onChange={setCpt} step={0.1} />
            <Field label="Chars/Word" value={cpw} onChange={setCpw} />
            <Field label="Words/Page" value={wpp} onChange={setWpp} />
            <Field label="Pages/Doc" value={ppd} onChange={setPpd} />
            <Field label="Tokens/Image" value={tpi} onChange={setTpi} />
          </div>
        </div>
      </div>

      {/* Derived Values */}
      <details style={{ marginBottom: 12 }}>
        <summary style={{ fontSize: 12, color: 'var(--text-dim)', cursor: 'pointer' }}>📈 Derived Values</summary>
        <div className="card" style={{ padding: 12, marginTop: 4 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 6 }}>
            <ResultCard label="Chars/Page" value={fmt(charsPerPage)} />
            <ResultCard label="Tokens/Page" value={fmt(tokensPerPage, 1)} />
            <ResultCard label="Words/Doc" value={fmt(wordsPerDoc)} />
            <ResultCard label="Chars/Doc" value={fmt(charsPerDoc)} />
            <ResultCard label="Tokens/Doc" value={fmt(tokensPerDoc, 1)} />
          </div>
        </div>
      </details>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
        {/* Cost Inputs */}
        <div className="card" style={{ padding: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 8 }}>🧮 Cost Estimation Inputs</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
            <Field label="Documents" value={docs} onChange={setDocs} />
            <Field label="Images" value={imgs} onChange={setImgs} />
          </div>
          <div style={{ marginTop: 8 }}>
            <label style={{ fontSize: 11, color: 'var(--text-dim)' }}>Output/Input Ratio: {ratio}</label>
            <input type="range" min="0.1" max="2" step="0.1" value={ratio}
              onChange={e => setRatio(Number(e.target.value))} style={{ width: '100%' }} />
          </div>
        </div>

        {/* Results */}
        <div className="card" style={{ padding: 12, border: '1px solid var(--accent)' }}>
          <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 8 }}>📋 Cost Outputs</div>
          {result ? (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 8 }}>
                <ResultCard label="Input Tokens" value={fmt(result.totalInput)} />
                <ResultCard label="Output Tokens" value={fmt(result.totalOutput)} />
                <ResultCard label="Input Cost" value={`$${result.inputCost.toFixed(4)}`} />
                <ResultCard label="Output Cost" value={`$${result.outputCost.toFixed(4)}`} />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
                <ResultCard label="Total Cost" value={`$${result.total.toFixed(4)}`} />
                <ResultCard label="Cost/Doc" value={`$${result.costPerDoc.toFixed(4)}`} />
                <ResultCard label="Cost/Page" value={`$${result.costPerPage.toFixed(6)}`} />
              </div>
            </>
          ) : (
            <div style={{ color: 'var(--text-dim)', fontSize: 12, textAlign: 'center', padding: 20 }}>
              Click Calculate to see results
            </div>
          )}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 8 }}>
        <button className="primary" onClick={calculate} style={{ flex: 1 }}>🔢 Calculate Cost</button>
        <button onClick={reset}>🔄 Reset</button>
      </div>
    </div>
  )
}

// ── Advanced Calculator Tab ──
function AdvancedCalculator({ config }) {
  const models = Object.values(config.models || {})
  const modelNames = models.map(m => m.name)
  const analyzers = config.analyzer_defaults || {}
  const analyzerNames = Object.keys(analyzers).sort()

  const [numPages, setNumPages] = useState(100)
  const [numDocs, setNumDocs] = useState(1)
  const [included, setIncluded] = useState(() => {
    const m = {}
    analyzerNames.forEach(n => m[n] = DEFAULT_INCLUDED.has(n))
    return m
  })
  const [modelOverrides, setModelOverrides] = useState(() => {
    const m = {}
    analyzerNames.forEach(n => m[n] = analyzers[n]?.default_model || 'Claude Sonnet 4.5')
    return m
  })
  const [result, setResult] = useState(null)

  const toggleInclude = (name) => setIncluded(p => ({ ...p, [name]: !p[name] }))
  const setAnalyzerModel = (name, val) => setModelOverrides(p => ({ ...p, [name]: val }))

  const getModelPricing = (name) => {
    const m = models.find(x => x.name === name) || models[0] || {}
    return [m.input_cost_per_million || 0, m.output_cost_per_million || 0]
  }

  const calculate = () => {
    const selected = analyzerNames.filter(n => included[n])
    const breakdown = []
    let totalInputTokens = 0, totalOutputTokens = 0
    let totalInputCost = 0, totalOutputCost = 0

    for (const name of selected) {
      const a = analyzers[name]
      const promptTokens = a.prompt_tokens || 0
      const outputTokens = a.expected_output_tokens || 0
      const inputPerPage = promptTokens + FIXED_IMAGE_TOKENS
      const analyzerInput = inputPerPage * numPages
      const analyzerOutput = outputTokens * numPages
      const [inpM, outM] = getModelPricing(modelOverrides[name])
      const inpCost = (analyzerInput / 1e6) * inpM
      const outCost = (analyzerOutput / 1e6) * outM

      totalInputTokens += analyzerInput
      totalOutputTokens += analyzerOutput
      totalInputCost += inpCost
      totalOutputCost += outCost

      breakdown.push({
        name, model: modelOverrides[name], promptTokens, imageTokens: FIXED_IMAGE_TOKENS,
        outputTokens, inputCost: inpCost, outputCost: outCost, totalCost: inpCost + outCost,
      })
    }

    const total = totalInputCost + totalOutputCost
    setResult({
      totalInputTokens, totalOutputTokens,
      inputCost: totalInputCost, outputCost: totalOutputCost, total,
      costPerPage: numPages > 0 ? total / numPages : 0,
      costPerDoc: numDocs > 0 ? total / numDocs : 0,
      selected: selected.length, breakdown,
    })
  }

  const reset = () => {
    setNumPages(100); setNumDocs(1); setResult(null)
    const inc = {}, mdl = {}
    analyzerNames.forEach(n => {
      inc[n] = DEFAULT_INCLUDED.has(n)
      mdl[n] = analyzers[n]?.default_model || 'Claude Sonnet 4.5'
    })
    setIncluded(inc); setModelOverrides(mdl)
  }

  return (
    <div>
      <div style={{ fontSize: 13, color: 'var(--text-dim)', marginBottom: 8 }}>
        Calculate costs based on actual deployed analyzer prompts. Toggle analyzers on/off and set per-analyzer models.
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 12, fontStyle: 'italic' }}>
        Image tokens fixed at {FIXED_IMAGE_TOKENS} (all images normalized to max 2048px)
      </div>

      <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
        <div className="card" style={{ padding: 12 }}>
          <Field label="Pages to Process" value={numPages} onChange={setNumPages} />
        </div>
        <div className="card" style={{ padding: 12 }}>
          <Field label="Documents (for per-doc cost)" value={numDocs} onChange={setNumDocs} />
        </div>
      </div>

      {/* Analyzer table */}
      <div className="card" style={{ padding: 12, marginBottom: 12 }}>
        <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 8 }}>🔧 Analyzer Configuration</div>
        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 8 }}>
          Toggle Include to add/remove analyzers. Change Model to override the default.
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '40px 1fr 1fr 80px 80px', gap: '4px 8px', alignItems: 'center', fontSize: 11 }}>
          <div style={{ fontWeight: 600, color: 'var(--text-dim)' }}>On</div>
          <div style={{ fontWeight: 600, color: 'var(--text-dim)' }}>Analyzer</div>
          <div style={{ fontWeight: 600, color: 'var(--text-dim)' }}>Model</div>
          <div style={{ fontWeight: 600, color: 'var(--text-dim)', textAlign: 'right' }}>Prompt</div>
          <div style={{ fontWeight: 600, color: 'var(--text-dim)', textAlign: 'right' }}>Output</div>
          {analyzerNames.map(name => {
            const a = analyzers[name]
            return [
              <input key={name + '-chk'} type="checkbox" checked={!!included[name]} onChange={() => toggleInclude(name)} />,
              <span key={name + '-name'} style={{ fontFamily: 'SF Mono, Menlo, monospace', fontSize: 11 }}>{name}</span>,
              <select key={name + '-mdl'} value={modelOverrides[name]} onChange={e => setAnalyzerModel(name, e.target.value)}
                style={{ fontSize: 11, padding: '2px 4px' }}>
                {modelNames.map(n => <option key={n} value={n}>{n}</option>)}
              </select>,
              <span key={name + '-pt'} style={{ textAlign: 'right', fontFamily: 'SF Mono, Menlo, monospace' }}>{fmt(a.prompt_tokens)}</span>,
              <span key={name + '-ot'} style={{ textAlign: 'right', fontFamily: 'SF Mono, Menlo, monospace' }}>{fmt(a.expected_output_tokens)}</span>,
            ]
          })}
        </div>
      </div>

      {/* Results */}
      <div className="card" style={{ padding: 12, marginBottom: 12, border: '1px solid var(--accent)' }}>
        <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 8 }}>📋 Cost Outputs</div>
        {result ? (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6, marginBottom: 8 }}>
              <ResultCard label="Input Tokens" value={fmt(result.totalInputTokens)} />
              <ResultCard label="Output Tokens" value={fmt(result.totalOutputTokens)} />
              <ResultCard label="Analyzers Selected" value={result.selected} />
              <ResultCard label="Input Cost" value={`$${result.inputCost.toFixed(4)}`} />
              <ResultCard label="Output Cost" value={`$${result.outputCost.toFixed(4)}`} />
              <ResultCard label="Total Cost" value={`$${result.total.toFixed(4)}`} />
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 12 }}>
              {result.selected} analyzers × {fmt(numPages)} pages = {fmt(result.totalInputTokens + result.totalOutputTokens)} total tokens.
              Cost/page: ${result.costPerPage.toFixed(6)} | Cost/doc: ${result.costPerDoc.toFixed(4)} | Total: ${result.total.toFixed(4)}
            </div>

            {/* Breakdown table */}
            <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 6 }}>📊 Analyzer Breakdown</div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    {['Analyzer', 'Model', 'Prompt', 'Image', 'Output', 'Input $', 'Output $', 'Total $'].map(h =>
                      <th key={h} style={{ padding: '4px 6px', textAlign: h === 'Analyzer' || h === 'Model' ? 'left' : 'right', color: 'var(--text-dim)' }}>{h}</th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {result.breakdown.map(b => (
                    <tr key={b.name} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={{ padding: '4px 6px', fontFamily: 'SF Mono, Menlo, monospace' }}>{b.name}</td>
                      <td style={{ padding: '4px 6px' }}>{b.model}</td>
                      <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'SF Mono, Menlo, monospace' }}>{fmt(b.promptTokens)}</td>
                      <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'SF Mono, Menlo, monospace' }}>{fmt(b.imageTokens)}</td>
                      <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'SF Mono, Menlo, monospace' }}>{fmt(b.outputTokens)}</td>
                      <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'SF Mono, Menlo, monospace' }}>${b.inputCost.toFixed(6)}</td>
                      <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'SF Mono, Menlo, monospace' }}>${b.outputCost.toFixed(6)}</td>
                      <td style={{ padding: '4px 6px', textAlign: 'right', fontFamily: 'SF Mono, Menlo, monospace' }}>${b.totalCost.toFixed(6)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <div style={{ color: 'var(--text-dim)', fontSize: 12, textAlign: 'center', padding: 20 }}>
            Click Calculate to see results
          </div>
        )}
      </div>

      <div style={{ display: 'flex', gap: 8 }}>
        <button className="primary" onClick={calculate} style={{ flex: 1 }}>🔢 Calculate Cost</button>
        <button onClick={reset}>🔄 Reset</button>
      </div>
    </div>
  )
}

// ── Main Component ──
export default function PricingCalculator() {
  const [config, setConfig] = useState(null)
  const [tab, setTab] = useState('basic')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/pricing-config')
      .then(r => r.json())
      .then(setConfig)
      .catch(() => setConfig(null))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="card" style={{ padding: 40, textAlign: 'center', color: 'var(--text-dim)' }}>Loading pricing config...</div>
  if (!config) return <div className="card" style={{ padding: 40, textAlign: 'center', color: 'var(--text-dim)' }}>Failed to load pricing config</div>

  return (
    <div>
      <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
        <button onClick={() => setTab('basic')}
          style={tab === 'basic' ? { background: '#1f6feb', borderColor: '#1f6feb' } : { fontSize: 12 }}>
          📊 Basic Calculator
        </button>
        <button onClick={() => setTab('advanced')}
          style={tab === 'advanced' ? { background: '#1f6feb', borderColor: '#1f6feb' } : { fontSize: 12 }}>
          🔬 Advanced Calculator
        </button>
      </div>
      {tab === 'basic' ? <BasicCalculator config={config} /> : <AdvancedCalculator config={config} />}
    </div>
  )
}
