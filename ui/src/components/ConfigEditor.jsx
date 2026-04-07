import React, { useState, useEffect } from 'react'

const TAG_LABELS = {
  application_name: 'Application Name',
  application_description: 'Description',
  environment: 'Environment',
  owner: 'Owner',
  cost_center: 'Cost Center',
  project_code: 'Project Code',
  cdk_stack_prefix: 'Stack Prefix',
  team: 'Team',
  team_contact_email: 'Team Email',
}

export default function ConfigEditor({ dirtyRef }) {
  const [tags, setTags] = useState({})
  const [savedTags, setSavedTags] = useState({})
  const [region, setRegion] = useState('')
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  const isDirty = JSON.stringify(tags) !== JSON.stringify(savedTags)

  useEffect(() => {
    if (dirtyRef) dirtyRef.current = isDirty
    return () => { if (dirtyRef) dirtyRef.current = false }
  }, [isDirty, dirtyRef])

  useEffect(() => {
    fetch('/api/config').then(r => r.json()).then(data => {
      setTags(data.tags || {})
      setSavedTags(data.tags || {})
      setRegion(data.region || '')
    })
  }, [])

  const save = async () => {
    setSaving(true)
    setSaved(false)
    await fetch('/api/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tags }),
    })
    setSavedTags({ ...tags })
    setSaving(false)
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const update = (key, value) => {
    setTags(prev => ({ ...prev, [key]: value }))
  }

  return (
    <div>
      <div style={{
        padding: 16, background: 'var(--surface)', borderRadius: 'var(--radius)',
        border: '1px solid var(--border)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <span style={{ fontSize: 15, fontWeight: 500 }}>Deployment Tags</span>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            {saved && <span style={{ fontSize: 12, color: 'var(--green)' }}>✓ Saved</span>}
            <button className="primary" onClick={save} disabled={saving}>
              {saving ? 'Saving...' : 'Save to app.py'}
            </button>
          </div>
        </div>

        <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 12, padding: '8px 12px', background: 'var(--bg)', borderRadius: 'var(--radius)' }}>
          Region: <span style={{ color: 'var(--accent)' }}>{region}</span> (set via CDK_DEFAULT_REGION env var)
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '180px 1fr', gap: '8px 12px', alignItems: 'center' }}>
          {Object.entries(tags).map(([key, value]) => (
            <React.Fragment key={key}>
              <label style={{ fontSize: 13, color: 'var(--text-dim)', textAlign: 'right' }}>
                {TAG_LABELS[key] || key}
              </label>
              <input
                value={value}
                onChange={e => update(key, e.target.value)}
                disabled={key === 'cdk_stack_prefix' || key === 'application_name'}
                style={key === 'cdk_stack_prefix' || key === 'application_name' ? { opacity: 0.5 } : {}}
              />
            </React.Fragment>
          ))}
        </div>

        <p style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 12 }}>
          These tags are applied to all AWS resources across all stacks. Changes are written directly to deployment/app.py.
        </p>
      </div>
    </div>
  )
}
