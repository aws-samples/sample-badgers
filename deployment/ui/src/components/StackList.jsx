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

export default function StackList({ runSSE, running }) {
  const [stacks, setStacks] = useState([])
  const [loading, setLoading] = useState(true)
  const [deploymentId, setDeploymentId] = useState('')
  const [expanded, setExpanded] = useState(null)
  const [outputs, setOutputs] = useState({})

  const fetchStacks = () => {
    setLoading(true)
    fetch('/api/stacks').then(r => r.json()).then(setStacks).finally(() => setLoading(false))
  }

  useEffect(() => { fetchStacks() }, [])
  // Refresh after a deploy/destroy finishes
  useEffect(() => { if (!running) fetchStacks() }, [running])

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
          <input
            placeholder="Deployment ID (optional)"
            value={deploymentId}
            onChange={e => setDeploymentId(e.target.value)}
            style={{ width: 200 }}
          />
          <button className="primary" disabled={running}
            onClick={() => {
              if (confirm('Deploy all stacks from scratch? This runs the full deploy_from_scratch.sh script.'))
                runSSE('/api/deploy-all', {})
            }}>
            🚀 Deploy All
          </button>
          <button onClick={() => {
              if (confirm('Sync local s3_files/ to the S3 config bucket?'))
                runSSE('/api/sync-s3', {})
            }} disabled={running}>
            ☁️ Sync S3
          </button>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {stacks.map(s => (
          <div key={s.id}>
            <div style={{
              display: 'grid', gridTemplateColumns: '1fr auto auto',
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
              <div style={{ display: 'flex', gap: 4 }}>
                <button disabled={running} style={{ fontSize: 12, padding: '3px 10px' }}
                  onClick={() => {
                    if (confirm(`Deploy ${s.stackName}?`))
                      runSSE('/api/deploy', { stackId: s.id, deploymentId })
                  }}>
                  Deploy
                </button>
                <button className="danger" disabled={running}
                  style={{ fontSize: 12, padding: '3px 10px' }}
                  onClick={() => {
                    if (confirm(`Destroy ${s.stackName}? This will permanently delete all resources in this stack.`))
                      runSSE('/api/destroy', { stackId: s.id })
                  }}>
                  Destroy
                </button>
              </div>
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
