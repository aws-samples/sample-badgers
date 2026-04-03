import React, { useRef, useEffect, useMemo } from 'react'

export default function LogPanel({ logs, onClear }) {
  const ref = useRef(null)
  const logText = useMemo(() => logs.join(''), [logs])

  // Parse "Step X/Y" from log lines for progress
  const progress = useMemo(() => {
    for (let i = logs.length - 1; i >= 0; i--) {
      const m = logs[i].match(/Step (\d+(?:\.\d+)?)\/(\d+)/)
      if (m) return { current: parseFloat(m[1]), total: parseInt(m[2]) }
    }
    return null
  }, [logs])

  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [logs])

  const pct = progress ? Math.round((progress.current / progress.total) * 100) : 0

  return (
    <div style={{
      marginTop: 24,
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius)',
      background: '#0d1117',
    }}>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '8px 12px', borderBottom: '1px solid var(--border)',
      }}>
        <span style={{ fontSize: 13, color: 'var(--text-dim)' }}>
          Output{progress ? ` — Step ${progress.current}/${progress.total} (${pct}%)` : ''}
        </span>
        <button onClick={onClear} style={{ fontSize: 11, padding: '2px 8px' }}>Clear</button>
      </div>
      {progress && (
        <div style={{ height: 4, background: '#161b22' }}>
          <div style={{
            height: '100%', width: `${pct}%`,
            background: pct === 100 ? 'var(--green, #3fb950)' : 'var(--accent, #1f6feb)',
            transition: 'width 0.4s ease',
          }} />
        </div>
      )}
      <pre ref={ref} style={{
        padding: 12, fontSize: 12, fontFamily: 'SF Mono, Menlo, monospace',
        maxHeight: 400, overflow: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-all',
        color: '#c9d1d9', lineHeight: 1.6,
      }}>
        {logText}
      </pre>
    </div>
  )
}
