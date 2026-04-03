import React, { useState, useRef } from 'react'
import Header from './components/Header.jsx'
import Home from './components/Home.jsx'
import Chat from './components/Chat.jsx'
import AnalyzerEditor from './components/AnalyzerEditor.jsx'
import AnalyzerWizard from './components/AnalyzerWizard.jsx'
import Evaluator from './components/Evaluator.jsx'
import PricingCalculator from './components/PricingCalculator.jsx'
import Observability from './components/Observability.jsx'
import ChatLog from './components/ChatLog.jsx'
import LogPanel from './components/LogPanel.jsx'

const TABS = [
  ['home', '🏠 Home'],
  ['chat', '💬 Chat'],
  ['editor', '✏️ Edit Analyzer'],
  ['wizard', '🧙 Create Analyzer'],
  ['evaluator', '🧪 Evaluations'],
  ['pricing', '💰 Pricing'],
  ['observability', '📊 Observability'],
  ['chatlog', '📝 Chat Log'],
]

export default function App() {
  const [tab, setTab] = useState('home')
  const [logs, setLogs] = useState([])
  const [running, setRunning] = useState(false)
  const dirtyRef = useRef(false)
  const abortRef = useRef(null)

  const MAX_LOG_LINES = 2000
  const appendLog = (line) => setLogs(prev => {
    const next = [...prev, line]
    return next.length > MAX_LOG_LINES ? next.slice(-MAX_LOG_LINES) : next
  })
  const clearLogs = () => setLogs([])

  const runSSE = (url, body) => {
    if (abortRef.current) abortRef.current.abort()
    const controller = new AbortController()
    abortRef.current = controller

    clearLogs()
    setRunning(true)
    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    }).then(async (res) => {
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
          try {
            const msg = JSON.parse(line.slice(6))
            if (msg.type === 'stdout' || msg.type === 'stderr') appendLog(msg.text)
            else if (msg.type === 'done') appendLog(msg.code === 0 ? '\n✅ Done' : `\n❌ Exit ${msg.code}`)
          } catch {}
        }
      }
    }).catch(e => {
      if (e.name === 'AbortError') return
      appendLog(`\n❌ ${e.message}`)
    }).finally(() => {
      abortRef.current = null
      setRunning(false)
    })
  }

  const switchTab = (t) => {
    if (dirtyRef.current && !confirm('Unsaved changes. Discard?')) return
    setTab(t)
  }

  return (
    <div className="container">
      <Header />
      <nav style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(([t, label]) => (
          <button key={t} onClick={() => switchTab(t)}
            style={tab === t ? { background: '#1f6feb', borderColor: '#1f6feb' } : { fontSize: 12 }}>
            {label}
          </button>
        ))}
      </nav>

      {tab === 'home' && <Home onNavigate={switchTab} />}
      <div style={{ display: tab === 'chat' ? 'block' : 'none' }}><Chat /></div>
      {tab === 'editor' && <AnalyzerEditor dirtyRef={dirtyRef} />}
      {tab === 'wizard' && <AnalyzerWizard runSSE={runSSE} running={running} />}
      {tab === 'evaluator' && <Evaluator />}
      {tab === 'pricing' && <PricingCalculator />}
      {tab === 'observability' && <Observability />}
      {tab === 'chatlog' && <ChatLog />}

      {logs.length > 0 && <LogPanel logs={logs} onClear={clearLogs} />}
    </div>
  )
}
