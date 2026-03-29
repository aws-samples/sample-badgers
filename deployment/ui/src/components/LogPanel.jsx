import React, { useRef, useEffect } from 'react'

export default function LogPanel({ logs, onClear }) {
  const ref = useRef(null)

  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [logs])

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
        <span style={{ fontSize: 13, color: 'var(--text-dim)' }}>Output</span>
        <button onClick={onClear} style={{ fontSize: 11, padding: '2px 8px' }}>Clear</button>
      </div>
      <pre ref={ref} style={{
        padding: 12, fontSize: 12, fontFamily: 'SF Mono, Menlo, monospace',
        maxHeight: 400, overflow: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-all',
        color: '#c9d1d9', lineHeight: 1.6,
      }}>
        {logs.join('')}
      </pre>
    </div>
  )
}
