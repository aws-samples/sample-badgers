import React, { useState, useEffect, useRef } from 'react'
import Header from './components/Header.jsx'
import ConfigEditor from './components/ConfigEditor.jsx'
import StackList from './components/StackList.jsx'
import S3ConfigEditor from './components/S3ConfigEditor.jsx'
import AnalyzerSelector from './components/AnalyzerSelector.jsx'
import LogPanel from './components/LogPanel.jsx'

export default function App() {
  const [tab, setTab] = useState('stacks')
  const [logs, setLogs] = useState([])
  const [running, setRunning] = useState(false)
  const dirtyRef = useRef(false)

  // Warn on browser close/refresh if running or unsaved
  useEffect(() => {
    const handler = (e) => {
      if (running || dirtyRef.current) {
        e.preventDefault()
        e.returnValue = ''
      }
    }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [running])

  const appendLog = (line) => setLogs(prev => [...prev, line])
  const clearLogs = () => setLogs([])

  const runSSE = (url, body) => {
    clearLogs()
    setRunning(true)
    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
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
            if (msg.type === 'stdout' || msg.type === 'stderr') {
              appendLog(msg.text)
            } else if (msg.type === 'done') {
              appendLog(msg.code === 0 ? '\n✅ Completed successfully' : `\n❌ Exited with code ${msg.code}`)
            }
          } catch {}
        }
      }
    }).catch(e => {
      appendLog(`\n❌ Error: ${e.message}`)
    }).finally(() => setRunning(false))
  }

  const switchTab = (t) => {
    if (dirtyRef.current && !confirm('You have unsaved changes. Discard them?')) return
    setTab(t)
  }

  return (
    <div className="container">
      <Header />
      <nav style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {[['stacks', '📦 Stacks'], ['analyzers', '🔬 Analyzers'], ['s3configs', '📄 S3 Configs'], ['config', '⚙️ Deploy Tags']].map(([t, label]) => (
          <button key={t} onClick={() => switchTab(t)}
            style={tab === t ? { background: '#1f6feb', borderColor: '#1f6feb' } : {}}>
            {label}
          </button>
        ))}
      </nav>

      {tab === 'stacks' && <StackList runSSE={runSSE} running={running} />}
      {tab === 'analyzers' && <AnalyzerSelector dirtyRef={dirtyRef} />}
      {tab === 's3configs' && <S3ConfigEditor runSSE={runSSE} running={running} dirtyRef={dirtyRef} />}
      {tab === 'config' && <ConfigEditor dirtyRef={dirtyRef} />}

      {logs.length > 0 && <LogPanel logs={logs} onClear={clearLogs} />}
    </div>
  )
}
