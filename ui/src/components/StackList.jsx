import React, { useState, useEffect } from 'react'

const STATUS_COLORS = {
  CREATE_COMPLETE: 'var(--green)',
  UPDATE_COMPLETE: 'var(--green)',
  UPDATE_ROLLBACK_COMPLETE: 'var(--yellow)',
  CREATE_IN_PROGRESS: 'var(--accent)',
  UPDATE_IN_PROGRESS: 'var(--accent)',
  ROLLBACK_IN_PROGRESS: 'var(--orange)',
  ROLLBACK_COMPLETE: 'var(--orange)',
  DELETE_IN_PROGRESS: 'var(--red)',
  CREATE_FAILED: 'var(--red)',
  DELETE_FAILED: 'var(--red)',
  NOT_DEPLOYED: 'var(--text-dim)',
  UNKNOWN: 'var(--text-dim)',
}

function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || 'var(--text-dim)'
  const label = status.replace(/_/g, ' ')
  return (
    <span style={{
      fontSize: 11, padding: '2px 8px', borderRadius: 12,
      border: `1px solid ${color}`, color, whiteSpace: 'nowrap',
    }}>
      {label}
    </span>
  )
}

export default function StackList({ running }) {
  const [stacks, setStacks] = useState([])
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState(null)
  const [outputs, setOutputs] = useState({})

  const fetchStacks = () => {
    setLoading(true)
    fetch('/api/stacks').then(r => r.json()).then(data => {
      setStacks(data)
      // Evict cached outputs for stacks that are no longer deployed
      setOutputs(prev => {
        const next = {}
        for (const s of data) {
          if (prev[s.id] && s.status.includes('COMPLETE') && !s.status.includes('DELETE')) {
            next[s.id] = prev[s.id]
          }
        }
        return next
      })
    }).finally(() => setLoading(false))
  }

  useEffect(() => { fetchStacks() }, [])
  // Refresh after a deploy/destroy finishes
  useEffect(() => { if (!running) fetchStacks() }, [running])
  // Poll stack status every 15s while a deploy is running
  useEffect(() => {
    if (!running) return
    const interval = setInterval(fetchStacks, 15000)
    return () => clearInterval(interval)
  }, [running])

  const toggleOutputs = async (stackId) => {
    if (expanded === stackId) { setExpanded(null); return }
    setExpanded(stackId)
    if (!outputs[stackId]) {
      try {
        const res = await fetch(`/api/stacks/${stackId}/outputs`)
        const data = await res.json()
        setOutputs(prev => ({ ...prev, [stackId]: data }))
      } catch {
        setOutputs(prev => ({ ...prev, [stackId]: [] }))
      }
    }
  }

  const deployedCount = stacks.filter(s =>
    s.status.includes('COMPLETE') && !s.status.includes('DELETE')
  ).length

  return (
    <div>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: 16, flexWrap: 'wrap', gap: 8,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 13, color: 'var(--text-dim)' }}>
            {deployedCount}/{stacks.length} deployed
          </span>
          <button onClick={fetchStacks} disabled={loading} style={{ fontSize: 12 }}>
            ↻ Refresh
          </button>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        </div>
      </div>

      <p style={{ fontSize: 12, color: 'var(--text-dim)', margin: '0 0 12px' }}>
        Read-only inventory for the deployment that hosts this application. Configure,
        deploy, resume, or destroy through the local deployment CLI so its reviewed
        per-deployment state is validated first.
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {stacks.map(s => (
          <div key={s.id}>
            <div style={{
              display: 'grid', gridTemplateColumns: '1fr auto',
              alignItems: 'center', gap: 12,
              padding: '10px 14px',
              background: 'var(--surface)',
              borderRadius: expanded === s.id ? 'var(--radius) var(--radius) 0 0' : 'var(--radius)',
              border: '1px solid var(--border)',
              borderBottom: expanded === s.id ? 'none' : undefined,
            }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <button onClick={() => toggleOutputs(s.id)}
                    style={{
                      background: 'none', border: 'none', padding: 0,
                      color: 'var(--text-dim)', fontSize: 11, width: 16,
                    }}>
                    {expanded === s.id ? '▼' : '▶'}
                  </button>
                  <span style={{ fontSize: 14, fontWeight: 500 }}>{s.name}</span>
                  <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>{s.stackName}</span>
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-dim)', marginLeft: 24, marginTop: 2 }}>
                  {s.description}
                </div>
              </div>
              <StatusBadge status={s.status} />
            </div>
            {expanded === s.id && (
              <div style={{
                padding: '10px 14px 10px 42px',
                background: 'var(--surface)',
                border: '1px solid var(--border)', borderTop: 'none',
                borderRadius: '0 0 var(--radius) var(--radius)',
                fontSize: 12,
              }}>
                {outputs[s.id]?.error ? (
                  <span style={{ color: 'var(--text-dim)' }}>{outputs[s.id].error}</span>
                ) : outputs[s.id]?.length > 0 ? (
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <tbody>
                      {outputs[s.id].map(o => (
                        <tr key={o.key}>
                          <td style={{ padding: '3px 12px 3px 0', color: 'var(--accent)', whiteSpace: 'nowrap' }}>{o.key}</td>
                          <td style={{ padding: '3px 0', fontFamily: 'SF Mono, Menlo, monospace', fontSize: 11, wordBreak: 'break-all' }}>{o.value}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <span style={{ color: 'var(--text-dim)' }}>No outputs</span>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
