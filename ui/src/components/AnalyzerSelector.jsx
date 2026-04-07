import { useState, useEffect } from 'react';

const CONTAINER_ANALYZERS = new Set(['image_enhancer', 'remediation_analyzer']);
const CATEGORY_ORDER = ['Document Processing', 'Visual Content', 'Text & Language', 'Structured Data', 'Metadata', 'Post-Processing'];

function label(name) {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function groupByCategory(analyzers) {
  const groups = {};
  for (const [name, cfg] of Object.entries(analyzers)) {
    const cat = cfg.category || 'Uncategorized';
    if (!groups[cat]) groups[cat] = [];
    groups[cat].push(name);
  }
  // Sort by defined order, unknowns at end
  const ordered = [];
  for (const cat of CATEGORY_ORDER) {
    if (groups[cat]) ordered.push([cat, groups[cat].sort()]);
  }
  for (const [cat, names] of Object.entries(groups)) {
    if (!CATEGORY_ORDER.includes(cat)) ordered.push([cat, names.sort()]);
  }
  return ordered;
}

export default function AnalyzerSelector({ dirtyRef }) {
  const [config, setConfig] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState(null);
  const [original, setOriginal] = useState(null);
  const [filter, setFilter] = useState('');

  useEffect(() => {
    fetch('/api/deployment-config').then(r => r.json()).then(data => {
      if (data.error) { setError(data.error); return; }
      setConfig(data);
      setOriginal(JSON.stringify(data));
    }).catch(e => setError(e.message));
  }, []);

  const isDirty = config && JSON.stringify(config) !== original;

  useEffect(() => {
    if (dirtyRef) dirtyRef.current = isDirty;
    return () => { if (dirtyRef) dirtyRef.current = false; };
  }, [isDirty, dirtyRef]);

  if (error) return <div style={{ color: 'var(--red)', padding: 20 }}>Error loading config: {error}</div>;
  if (!config) return <div style={{ color: 'var(--text-dim)', padding: 20 }}>Loading...</div>;

  const analyzers = config.analyzers || {};
  const names = Object.keys(analyzers);
  const enabledCount = names.filter(n => analyzers[n].enabled).length;
  const filterKey = filter.toLowerCase().replace(/ /g, '_');
  const groups = groupByCategory(analyzers);

  const toggle = (name) => {
    setConfig(prev => ({
      ...prev,
      analyzers: { ...prev.analyzers, [name]: { ...prev.analyzers[name], enabled: !prev.analyzers[name].enabled } }
    }));
    setSaved(false);
  };

  const setAll = (enabled) => {
    setConfig(prev => {
      const updated = { ...prev.analyzers };
      for (const n of names) updated[n] = { ...updated[n], enabled };
      return { ...prev, analyzers: updated };
    });
    setSaved(false);
  };

  const setCategoryAll = (category, enabled) => {
    setConfig(prev => {
      const updated = { ...prev.analyzers };
      for (const [n, cfg] of Object.entries(updated)) {
        if ((cfg.category || 'Uncategorized') === category) updated[n] = { ...cfg, enabled };
      }
      return { ...prev, analyzers: updated };
    });
    setSaved(false);
  };

  const save = async () => {
    setSaving(true);
    try {
      await fetch('/api/deployment-config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      setSaved(true);
      setOriginal(JSON.stringify(config));
      setTimeout(() => setSaved(false), 2000);
    } catch (e) { setError(e.message); }
    setSaving(false);
  };

  return (
    <div>
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: 12, flexWrap: 'wrap', gap: 8,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 13, color: 'var(--text-dim)' }}>
            {enabledCount}/{names.length} enabled
          </span>
          <input
            placeholder="Filter analyzers..."
            value={filter}
            onChange={e => setFilter(e.target.value)}
            style={{ width: 200, fontSize: 12 }}
          />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button onClick={() => setAll(true)} style={{ fontSize: 12 }}>Enable All</button>
          <button onClick={() => setAll(false)} style={{ fontSize: 12 }}>Disable All</button>
          {saved && <span style={{ fontSize: 11, color: 'var(--green)' }}>✓ Saved</span>}
          {isDirty && <span style={{ fontSize: 11, color: 'var(--yellow)' }}>Unsaved changes</span>}
          <button className="primary" onClick={save} disabled={saving || !isDirty}>
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>

      <div style={{
        padding: '10px 14px', marginBottom: 12,
        background: 'var(--surface)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius)', fontSize: 12, color: 'var(--text-dim)',
      }}>
        Toggle analyzers on/off to control which Lambda functions get deployed.
        Disabled analyzers won't create Lambda functions or Gateway targets.
        Save changes here, then redeploy the Lambda + Gateway stacks.
      </div>

      {groups.map(([category, catNames]) => {
        const visible = filter ? catNames.filter(n => n.includes(filterKey)) : catNames;
        if (visible.length === 0) return null;
        const catEnabled = visible.filter(n => analyzers[n].enabled).length;
        return (
          <div key={category} style={{ marginBottom: 16 }}>
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              marginBottom: 6, padding: '0 4px',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 13, fontWeight: 600 }}>{category}</span>
                <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                  {catEnabled}/{visible.length}
                </span>
              </div>
              <div style={{ display: 'flex', gap: 4 }}>
                <button onClick={() => setCategoryAll(category, true)}
                  style={{ fontSize: 11, padding: '2px 8px', background: 'none', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-dim)', cursor: 'pointer' }}>
                  all on
                </button>
                <button onClick={() => setCategoryAll(category, false)}
                  style={{ fontSize: 11, padding: '2px 8px', background: 'none', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-dim)', cursor: 'pointer' }}>
                  all off
                </button>
              </div>
            </div>
            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
              gap: 4,
            }}>
              {visible.map(name => {
                const enabled = analyzers[name].enabled;
                const isContainer = CONTAINER_ANALYZERS.has(name);
                return (
                  <div key={name} onClick={() => toggle(name)} style={{
                    display: 'flex', alignItems: 'center', gap: 10,
                    padding: '8px 12px', cursor: 'pointer',
                    background: enabled ? 'rgba(31,111,235,0.08)' : 'var(--surface)',
                    border: `1px solid ${enabled ? 'rgba(31,111,235,0.3)' : 'var(--border)'}`,
                    borderRadius: 'var(--radius)',
                    opacity: enabled ? 1 : 0.6,
                    transition: 'all 0.15s',
                  }}>
                    <div style={{
                      width: 36, height: 20, borderRadius: 10,
                      background: enabled ? 'var(--accent)' : 'var(--border)',
                      position: 'relative', transition: 'background 0.15s', flexShrink: 0,
                    }}>
                      <div style={{
                        width: 16, height: 16, borderRadius: 8,
                        background: '#fff', position: 'absolute', top: 2,
                        left: enabled ? 18 : 2, transition: 'left 0.15s',
                      }} />
                    </div>
                    <div style={{ minWidth: 0 }}>
                      <div style={{ fontSize: 13, fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {label(name)}
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                        {name}{isContainer ? ' (container)' : ''}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
