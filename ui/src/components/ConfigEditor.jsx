import { useEffect } from 'react'

export default function ConfigEditor({ dirtyRef }) {
  useEffect(() => {
    if (dirtyRef) dirtyRef.current = false
    return () => { if (dirtyRef) dirtyRef.current = false }
  }, [dirtyRef])

  return (
    <div style={{
      padding: 16, background: 'var(--surface)', borderRadius: 'var(--radius)',
      border: '1px solid var(--border)', maxWidth: 720,
    }}>
      <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 10 }}>
        Deployment tag configuration
      </div>

      <p style={{ fontSize: 13, color: 'var(--text-dim)', margin: '0 0 14px', lineHeight: 1.55 }}>
        Tags are configured per deployment ID in the local deployment planner. This
        application view is intentionally read-only: it does not modify source files,
        local desired state, or AWS resources.
      </p>

      <div style={{
        fontFamily: 'SF Mono, Menlo, monospace', fontSize: 13, padding: '10px 12px',
        background: 'var(--bg)', borderRadius: 'var(--radius)', marginBottom: 14,
      }}>
        ./deploy.sh configure --ui
      </div>

      <ol style={{ fontSize: 13, color: 'var(--text-dim)', margin: 0, paddingLeft: 22, lineHeight: 1.7 }}>
        <li>Open an existing deployment ID or explicitly create a new local configuration.</li>
        <li>Choose resource tags and review the resulting deployment plan.</li>
        <li>Run the displayed resume command and select the same deployment ID in the CLI.</li>
      </ol>

      <p style={{ fontSize: 12, color: 'var(--text-dim)', margin: '14px 0 0' }}>
        Tag changes are reconciled across every enabled stack when the CLI deployment runs.
      </p>
    </div>
  )
}
