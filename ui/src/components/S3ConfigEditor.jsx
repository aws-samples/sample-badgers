import { useState, useEffect, useRef } from 'react';
import JsonHighlighter, { THEMES } from './JsonHighlighter.jsx';

function groupByDir(files) {
  const groups = {};
  for (const f of files) {
    const parts = f.split('/');
    const dir = parts.length > 1 ? parts.slice(0, -1).join('/') : '(root)';
    if (!groups[dir]) groups[dir] = [];
    groups[dir].push(f);
  }
  return groups;
}

const DIR_DESC = {
  'agent_config': 'Agent model, temperature, thinking mode, and operating environment guardrails',
  'manifests': 'Per-analyzer config: model selections, extended thinking budgets, fallback chains',
  'schemas': 'JSON schemas defining input/output contracts for each analyzer tool',
  'prompts': 'System and user prompt templates that drive each analyzer',
  'core_system_prompts': 'Shared prompt rules: audit mode, error handling, core extraction rules',
  'agent_system_prompt': 'Top-level system prompt for the Strands orchestration agent',
  'wrappers': 'XML wrapper templates that compose the final prompt sent to models',
};

export default function S3ConfigEditor({ runSSE, running, dirtyRef }) {
  const [files, setFiles] = useState([]);
  const [selected, setSelected] = useState(null);
  const [editText, setEditText] = useState('');
  const [lastSavedText, setLastSavedText] = useState('');
  const [parseError, setParseError] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem('hljs-theme') || 'github-dark');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [expandedDirs, setExpandedDirs] = useState({});
  const textareaRef = useRef(null);

  useEffect(() => {
    fetch('/api/s3-configs').then(r => r.json()).then(data => {
      setFiles(data);
      const dirs = {};
      for (const f of data) {
        const p = f.split('/');
        dirs[p.length > 1 ? p.slice(0, -1).join('/') : '(root)'] = true;
      }
      setExpandedDirs(dirs);
    });
  }, []);

  const isDirty = editText !== lastSavedText && selected;

  // Keep parent dirtyRef in sync
  useEffect(() => {
    if (dirtyRef) dirtyRef.current = isDirty;
    return () => { if (dirtyRef) dirtyRef.current = false; };
  }, [isDirty, dirtyRef]);

  const loadAbortRef = useRef(null);

  const loadFile = async (path) => {
    if (isDirty && !confirm('You have unsaved changes. Discard them?')) return;
    // Abort any in-flight load
    if (loadAbortRef.current) loadAbortRef.current.abort();
    const controller = new AbortController();
    loadAbortRef.current = controller;

    setSelected(path); setSaved(false); setParseError(null);
    try {
      const res = await fetch('/api/s3-configs/' + path, { signal: controller.signal });
      const data = await res.json();
      const text = data.error ? '' : JSON.stringify(data.content, null, 4);
      setEditText(text);
      setLastSavedText(text);
    } catch (e) {
      if (e.name !== 'AbortError') {
        setEditText('');
        setLastSavedText('');
      }
    }
  };

  const handleEdit = (val) => {
    setEditText(val); setSaved(false);
    try { JSON.parse(val); setParseError(null); } catch (e) { setParseError(e.message); }
  };

  const save = async () => {
    if (parseError) return;
    setSaving(true);
    await fetch('/api/s3-configs/' + selected, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: JSON.parse(editText) }),
    });
    setSaving(false); setSaved(true); setLastSavedText(editText);
    setTimeout(() => setSaved(false), 2000);
  };

  const toggleDir = (dir) => setExpandedDirs(p => ({ ...p, [dir]: !p[dir] }));
  const groups = groupByDir(files);
  const fileName = selected ? selected.split('/').pop() : null;
  const SW = 280;
  const CW = 36;

  const dirDescPanel = (
    <div style={{
      marginBottom: 12, padding: '10px 14px',
      background: 'var(--surface)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius)', fontSize: 12, color: 'var(--text-dim)',
      display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 24px',
    }}>
      {Object.entries(DIR_DESC).map(([dir, desc]) => (
        <div key={dir} style={{ display: 'flex', gap: 6 }}>
          <span style={{ color: 'var(--accent)', whiteSpace: 'nowrap' }}>{dir}/</span>
          <span>{desc}</span>
        </div>
      ))}
    </div>
  );

  const sidebar = (
    <div style={{
      background: 'var(--surface)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius) 0 0 var(--radius)',
      overflow: 'hidden', display: 'flex', flexDirection: 'column',
      width: sidebarOpen ? SW : CW, transition: 'width 0.2s',
    }}>
      <div style={{
        padding: sidebarOpen ? '8px 8px 8px 12px' : '8px',
        borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center',
        justifyContent: sidebarOpen ? 'space-between' : 'center',
        minHeight: 36,
      }}>
        {sidebarOpen && <span style={{ fontSize: 13, color: 'var(--text-dim)' }}>s3_files/</span>}
        <button
          onClick={() => setSidebarOpen(p => !p)}
          style={{ background: 'none', border: 'none', padding: '2px 4px', color: 'var(--text-dim)', fontSize: 14, lineHeight: 1 }}
          title={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
        >{sidebarOpen ? '\u25C0' : '\u25B6'}</button>
      </div>
      {sidebarOpen && (
        <div style={{ overflow: 'auto', flex: 1 }}>
          {Object.entries(groups).map(([dir, dirFiles]) => (
            <div key={dir}>
              <div onClick={() => toggleDir(dir)} style={{
                padding: '6px 12px', fontSize: 11, color: 'var(--text-dim)',
                background: 'var(--bg)', cursor: 'pointer', userSelect: 'none',
                display: 'flex', alignItems: 'center', gap: 6,
              }}>
                <span style={{ fontSize: 9, width: 10, textAlign: 'center' }}>
                  {expandedDirs[dir] ? '\u25BC' : '\u25B6'}
                </span>
                {'\uD83D\uDCC1'} {dir}
                <span style={{ marginLeft: 'auto', fontSize: 10, opacity: 0.6 }}>{dirFiles.length}</span>
              </div>
              {expandedDirs[dir] && dirFiles.map(f => {
                const name = f.split('/').pop();
                const isActive = f === selected;
                return (
                  <div key={f} onClick={() => loadFile(f)} style={{
                    padding: '5px 12px 5px 28px', fontSize: 12, cursor: 'pointer',
                    background: isActive ? 'var(--accent-subtle)' : 'transparent',
                    color: isActive ? 'var(--accent)' : 'var(--text)',
                    borderLeft: isActive ? '2px solid var(--accent)' : '2px solid transparent',
                  }}>{name}</div>
                );
              })}
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const editor = (
    <div style={{
      background: 'var(--surface)', border: '1px solid var(--border)', borderLeft: 'none',
      borderRadius: '0 var(--radius) var(--radius) 0',
      display: 'flex', flexDirection: 'column',
    }}>
      {selected ? (
        <>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '8px 12px', borderBottom: '1px solid var(--border)', flexWrap: 'wrap', gap: 6,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 13, fontWeight: 500 }}>{fileName}</span>
              <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>{selected}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <select value={theme} onChange={e => { setTheme(e.target.value); localStorage.setItem('hljs-theme', e.target.value); }} style={{ width: 160, fontSize: 12 }}>
                {THEMES.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
              {parseError && <span style={{ fontSize: 11, color: 'var(--red)' }}>Invalid JSON</span>}
              {saved && <span style={{ fontSize: 11, color: 'var(--green)' }}>{'\u2713'} Saved</span>}
              <button className="primary" onClick={save} disabled={saving || !!parseError}>{saving ? 'Saving...' : 'Save'}</button>
              <button onClick={() => { if (confirm('Sync local s3_files/ to the S3 config bucket?')) runSSE('/api/sync-s3', {}); }} disabled={running} style={{ fontSize: 12 }}>{'\u2601\uFE0F'} Sync to S3</button>
            </div>
          </div>
          <div style={{
            flex: 1, position: 'relative', overflow: 'auto', background: '#0d1117',
            borderTop: parseError ? '2px solid var(--red)' : '2px solid transparent',
          }}>
            <div style={{ position: 'absolute', top: 0, left: 0, right: 0, minHeight: '100%' }}>
              <JsonHighlighter text={editText} theme={theme} />
            </div>
            <textarea
              ref={textareaRef}
              value={editText}
              onChange={e => handleEdit(e.target.value)}
              spellCheck={false}
              style={{
                position: 'relative', width: '100%', height: '100%', minHeight: 400,
                background: 'transparent', color: 'transparent', caretColor: '#c9d1d9',
                border: 'none', padding: 12,
                fontFamily: 'SF Mono, Menlo, monospace', fontSize: 13,
                lineHeight: 1.6, resize: 'none', outline: 'none',
                whiteSpace: 'pre-wrap', wordBreak: 'break-all',
              }}
            />
          </div>
        </>
      ) : (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)', fontSize: 13 }}>
          Select a config file to edit
        </div>
      )}
    </div>
  );

  return (
    <div>
      {dirDescPanel}
      <div style={{
        display: 'grid',
        gridTemplateColumns: (sidebarOpen ? SW : CW) + 'px 1fr',
        gap: 0, minHeight: 500,
        transition: 'grid-template-columns 0.2s',
      }}>
        {sidebar}
        {editor}
      </div>
    </div>
  );
}