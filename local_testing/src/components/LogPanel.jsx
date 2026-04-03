import { useRef, useEffect, useMemo } from 'react'

export default function LogPanel({ logs, onClear }) {
  const ref = useRef(null)
  const logText = useMemo(() => logs.join(''), [logs])
  useEffect(() => { ref.current?.scrollTo(0, ref.current.scrollHeight) }, [logs])

  return (
    <div style={{ marginTop: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <span style={{ fontSize: 13, color: 'var(--text-dim)' }}>Output Log</span>
        <button onClick={onClear} style={{ fontSize: 11, padding: '2px 8px' }}>Clear</button>
      </div>
      <div ref={ref} className="card" style={{
        padding: 12, maxHeight: 300, overflow: 'auto',
        fontFamily: 'SF Mono, Menlo, monospace', fontSize: 12,
        whiteSpace: 'pre-wrap', wordBreak: 'break-word',
      }}>
        {logText}
      </div>
    </div>
  )
}
