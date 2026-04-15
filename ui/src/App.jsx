import React, { useState, useEffect, useRef } from 'react'
import { useUser } from './hooks/useUser.js'
import Header from './components/Header.jsx'
import Home from './components/Home.jsx'
import Chat from './components/Chat.jsx'
import AnalyzerEditor from './components/AnalyzerEditor.jsx'
import AnalyzerWizard from './components/AnalyzerWizard.jsx'
import Evaluator from './components/Evaluator.jsx'
import PricingCalculator from './components/PricingCalculator.jsx'
import Observability from './components/Observability.jsx'
import ChatLog from './components/ChatLog.jsx'
import StackList from './components/StackList.jsx'
import AnalyzerSelector from './components/AnalyzerSelector.jsx'
import S3ConfigEditor from './components/S3ConfigEditor.jsx'
import ConfigEditor from './components/ConfigEditor.jsx'
import LogPanel from './components/LogPanel.jsx'

const TABS = [
    // Testing tabs — all roles
    { id: 'home', label: '🏠 Home' },
    { id: 'chat', label: '💬 Chat' },
    { id: 'editor', label: '✏️ Edit Analyzer' },
    { id: 'wizard', label: '🧙 Create Analyzer' },
    { id: 'evaluator', label: '🧪 Evaluations' },
    { id: 'pricing', label: '💰 Pricing' },
    { id: 'observability', label: '📊 Observability' },
    { id: 'chatlog', label: '📝 Chat Log' },
    // Admin tabs
    { id: 'stacks', label: '📦 Stacks', adminOnly: true },
    { id: 'analyzers', label: '🔬 Analyzers', adminOnly: true },
    { id: 's3configs', label: '📄 S3 Configs', adminOnly: true },
    { id: 'config', label: '⚙️ Deploy Tags', adminOnly: true },
]

export default function App() {
    const { role, loading } = useUser()
    const [tab, setTab] = useState('home')
    const [logs, setLogs] = useState([])
    const [running, setRunning] = useState(false)
    const [branding, setBranding] = useState({})
    const [theme, setTheme] = useState(() => localStorage.getItem('badgers-theme') || '')
    const dirtyRef = useRef(false)
    const abortRef = useRef(null)

    function applyTheme(t) {
        setTheme(t)
        localStorage.setItem('badgers-theme', t)
        document.documentElement.setAttribute('data-theme', t)
    }

    // Fetch branding from /api/env (retry if backend isn't ready yet)
    useEffect(() => {
        let cancelled = false
        async function fetchEnv(retries = 10, delay = 500) {
            for (let i = 0; i < retries; i++) {
                if (cancelled) return
                try {
                    const res = await fetch('/api/env')
                    if (!res.ok) throw new Error(res.status)
                    const data = await res.json()
                    if (data.branding) {
                        setBranding(data.branding)
                        document.title = data.branding.appName || 'BADGERS'
                        const saved = localStorage.getItem('badgers-theme')
                        applyTheme(saved || data.branding.theme || 'dark')
                    }
                    return
                } catch {
                    await new Promise(r => setTimeout(r, delay))
                }
            }
        }
        fetchEnv()
        return () => { cancelled = true }
    }, [])

    const testingTabs = TABS.filter(t => !t.adminOnly)
    const adminTabs = TABS.filter(t => t.adminOnly)
    const showAdmin = role === 'admin'

    // Warn on browser close if running or unsaved
    useEffect(() => {
        const handler = (e) => { if (running || dirtyRef.current) { e.preventDefault(); e.returnValue = ''; } }
        window.addEventListener('beforeunload', handler)
        return () => window.removeEventListener('beforeunload', handler)
    }, [running])

    useEffect(() => {
        return () => { if (abortRef.current) abortRef.current.abort() }
    }, [])

    const MAX_LOG_LINES = 2000
    const appendLog = (line) => setLogs(prev => {
        const next = [...prev, line]
        return next.length > MAX_LOG_LINES ? next.slice(-MAX_LOG_LINES) : next
    })
    const clearLogs = () => setLogs([])

    const parseSSELine = (line) => {
        if (!line.startsWith('data: ')) return
        try {
            const msg = JSON.parse(line.slice(6))
            if (msg.type === 'stdout' || msg.type === 'stderr') appendLog(msg.text)
            else if (msg.type === 'done') appendLog(msg.code === 0 ? '\n✅ Completed successfully' : `\n❌ Exited with code ${msg.code}`)
        } catch {}
    }

    const runSSE = (url, body) => {
        if (abortRef.current) abortRef.current.abort()
        const controller = new AbortController()
        abortRef.current = controller
        clearLogs()
        setRunning(true)
        fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: controller.signal })
            .then(async (res) => {
                const reader = res.body.getReader()
                const decoder = new TextDecoder()
                let buffer = ''
                while (true) {
                    const { done, value } = await reader.read()
                    if (done) break
                    buffer += decoder.decode(value, { stream: true })
                    const lines = buffer.split('\n')
                    buffer = lines.pop()
                    for (const line of lines) parseSSELine(line)
                }
            })
            .catch(e => { if (e.name === 'AbortError') return; appendLog(`\n❌ Error: ${e.message}`) })
            .finally(() => { abortRef.current = null; setRunning(false) })
    }

    const runSSEGet = (url) => {
        if (abortRef.current) abortRef.current.abort()
        const controller = new AbortController()
        abortRef.current = controller
        clearLogs()
        setRunning(true)
        fetch(url, { signal: controller.signal })
            .then(async (res) => {
                const reader = res.body.getReader()
                const decoder = new TextDecoder()
                let buffer = ''
                while (true) {
                    const { done, value } = await reader.read()
                    if (done) break
                    buffer += decoder.decode(value, { stream: true })
                    const lines = buffer.split('\n')
                    buffer = lines.pop()
                    for (const line of lines) parseSSELine(line)
                }
            })
            .catch(e => { if (e.name === 'AbortError') return; appendLog(`\n❌ Error: ${e.message}`) })
            .finally(() => { abortRef.current = null; setRunning(false) })
    }

    const switchTab = (t) => {
        if (dirtyRef.current && !confirm('You have unsaved changes. Discard them?')) return
        setTab(t)
    }

    if (loading) return <div className="container" style={{ paddingTop: 60, textAlign: 'center', color: 'var(--text-dim)' }}>Loading...</div>

    return (
        <div className="container">
            <Header branding={branding} theme={theme} onThemeChange={applyTheme} />
            <nav style={{ marginBottom: 20 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, flexWrap: 'wrap' }}>
                    <span style={{ fontSize: 11, color: 'var(--text-dim)', width: 52, flexShrink: 0 }}>Testing</span>
                    {testingTabs.map(({ id, label }) => (
                        <button key={id} onClick={() => switchTab(id)}
                            className={tab === id ? 'tab-active' : ''}
                            style={{ fontSize: 12 }}>
                            {label}
                        </button>
                    ))}
                </div>
                {showAdmin && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                        <span style={{ fontSize: 11, color: 'var(--text-dim)', width: 52, flexShrink: 0 }}>Deploy</span>
                        {adminTabs.map(({ id, label }) => (
                            <button key={id} onClick={() => switchTab(id)}
                                className={tab === id ? 'tab-active' : ''}
                                style={{ fontSize: 12 }}>
                                {label}
                            </button>
                        ))}
                    </div>
                )}
            </nav>

            {/* Testing tabs */}
            {tab === 'home' && <Home onNavigate={switchTab} branding={branding} />}
            <div style={{ display: tab === 'chat' ? 'block' : 'none' }}><Chat /></div>
            {tab === 'editor' && <AnalyzerEditor dirtyRef={dirtyRef} />}
            {tab === 'wizard' && <AnalyzerWizard runSSE={runSSE} running={running} />}
            {tab === 'evaluator' && <Evaluator />}
            {tab === 'pricing' && <PricingCalculator />}
            {tab === 'observability' && <Observability />}
            {tab === 'chatlog' && <ChatLog />}

            {/* Admin tabs */}
            {role === 'admin' && tab === 'stacks' && <StackList runSSE={runSSE} runSSEGet={runSSEGet} running={running} />}
            {role === 'admin' && tab === 'analyzers' && <AnalyzerSelector dirtyRef={dirtyRef} />}
            {role === 'admin' && tab === 's3configs' && <S3ConfigEditor runSSE={runSSE} running={running} dirtyRef={dirtyRef} />}
            {role === 'admin' && tab === 'config' && <ConfigEditor dirtyRef={dirtyRef} />}

            {logs.length > 0 && <LogPanel logs={logs} onClear={clearLogs} />}
        </div>
    )
}
