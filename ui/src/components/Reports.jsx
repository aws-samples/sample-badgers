import React, { useCallback, useEffect, useMemo, useState } from 'react'
import ShikiHighlighter from 'react-shiki'
import CopyButton from './CopyButton'

const VIEW_TABS = [
  ['overview', '▦ Overview'],
  ['reader', '◫ Page Reader'],
  ['audit', '✓ Audit Trail'],
]

function formatDate(value) {
  if (!value) return '—'
  try { return new Date(value).toLocaleString() } catch { return value }
}

function duration(start, end) {
  if (!start || !end) return '—'
  const ms = new Date(end).getTime() - new Date(start).getTime()
  if (!Number.isFinite(ms) || ms < 0) return '—'
  return ms < 60000 ? `${Math.round(ms / 1000)}s` : `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s`
}

export default function Reports() {
  const [reports, setReports] = useState([])
  const [reportId, setReportId] = useState('')
  const [manifest, setManifest] = useState(null)
  const [loadedReportId, setLoadedReportId] = useState('')
  const [view, setView] = useState('overview')
  const [pageIndex, setPageIndex] = useState(0)
  const [spineTab, setSpineTab] = useState('rendered')
  const [imageUrl, setImageUrl] = useState('')
  // Empty unless the run enhanced this page. Drives whether the image pane offers
  // tabs at all rather than showing an empty "Enhanced" view.
  const [enhancedImageUrl, setEnhancedImageUrl] = useState('')
  const [imageTab, setImageTab] = useState('original')
  const [xml, setXml] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const currentPage = manifest?.pages?.[pageIndex] || null

  const refresh = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const response = await fetch('/api/reports')
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Could not load reports')
      const next = data.reports || []
      setReports(next)
      setReportId(previous => previous && next.some(item => item.report_id === previous) ? previous : (next[0]?.report_id || ''))
    } catch (e) {
      setReports([])
      setError(e.message)
    }
    setLoading(false)
  }, [])

  useEffect(() => { refresh() }, [refresh])

  useEffect(() => {
    if (!reportId) {
      setManifest(null)
      setLoadedReportId('')
      setImageUrl('')
      setXml('')
      return
    }
    let cancelled = false
    setManifest(null)
    setLoadedReportId('')
    setImageUrl('')
    setXml('')
    setLoading(true)
    setError('')
    fetch(`/api/reports/${encodeURIComponent(reportId)}/manifest`)
      .then(async response => {
        const data = await response.json()
        if (!response.ok) throw new Error(data.error || 'Could not load report')
        if (!cancelled) {
          setManifest(data)
          setLoadedReportId(reportId)
          setPageIndex(0)
          setView('overview')
        }
      })
      .catch(e => { if (!cancelled) setError(e.message) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [reportId])

  useEffect(() => {
    if (!reportId || !currentPage || loadedReportId !== reportId) {
      setImageUrl('')
      setEnhancedImageUrl('')
      setXml('')
      return
    }
    setImageUrl('')
    setEnhancedImageUrl('')
    setImageTab('original')
    setXml('')
    let cancelled = false
    const objectUrls = []
    const base = `/api/reports/${encodeURIComponent(reportId)}/pages/${encodeURIComponent(currentPage.page_number)}`
    // Only requested when the manifest declares one, so an un-enhanced page (or a
    // report generated before enhanced copies were kept) costs no extra request.
    const hasEnhanced = Boolean(currentPage.enhanced_image_key)
    Promise.all([
      fetch(`${base}/image`).then(async response => {
        if (!response.ok) throw new Error('Could not load page image')
        return response.blob()
      }),
      fetch(`${base}/xml`).then(async response => {
        if (!response.ok) throw new Error('Could not load page spine')
        return response.text()
      }),
      // Supplementary: resolves to null on failure so a missing enhanced copy
      // cannot stop the page from rendering.
      hasEnhanced
        ? fetch(`${base}/enhanced-image`)
            .then(response => (response.ok ? response.blob() : null))
            .catch(() => null)
        : Promise.resolve(null),
    ]).then(([blob, pageXml, enhancedBlob]) => {
      if (cancelled) return
      const url = URL.createObjectURL(blob)
      objectUrls.push(url)
      setImageUrl(url)
      setXml(pageXml)
      if (enhancedBlob) {
        const enhancedUrl = URL.createObjectURL(enhancedBlob)
        objectUrls.push(enhancedUrl)
        setEnhancedImageUrl(enhancedUrl)
      }
    }).catch(e => { if (!cancelled) setError(e.message) })
    return () => {
      cancelled = true
      for (const url of objectUrls) URL.revokeObjectURL(url)
    }
  }, [reportId, currentPage, loadedReportId])

  useEffect(() => {
    const handler = event => {
      if (view !== 'reader' || !manifest?.pages?.length) return
      if (event.key === 'ArrowLeft') setPageIndex(index => Math.max(0, index - 1))
      if (event.key === 'ArrowRight') setPageIndex(index => Math.min(manifest.pages.length - 1, index + 1))
      if (event.key === 'Home') setView('overview')
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [view, manifest])

  const pageSpecialists = useMemo(() => currentPage?.specialists || [], [currentPage])
  const documentSpine = useMemo(() => {
    if (!manifest?.pages) return ''
    return `<document_spine job_id="${manifest.job_id || manifest.report_id}" pages="${manifest.pages.length}">\n${manifest.pages.map(page => `  <page number="${page.page_number}" source="${page.spine_key}">\n    <summary>${page.summary || ''}</summary>\n  </page>`).join('\n')}\n</document_spine>`
  }, [manifest])

  const openPage = index => {
    setPageIndex(index)
    setSpineTab('rendered')
    setView('reader')
  }

  const download = async () => {
    if (!reportId || loadedReportId !== reportId) return
    try {
      const response = await fetch(`/api/reports/${encodeURIComponent(reportId)}/download`)
      if (!response.ok) throw new Error('Could not download report')
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${reportId}.html`
      link.click()
      URL.revokeObjectURL(url)
    } catch (e) { setError(e.message) }
  }

  if (loading && !manifest && reports.length === 0) {
    return <div className="card report-empty">Loading reports...</div>
  }

  return (
    <div className="reports-root">
      <div className="report-toolbar">
        <select value={reportId} onChange={event => setReportId(event.target.value)}>
          <option value="">Select a completed report...</option>
          {reports.map(report => <option key={report.report_id} value={report.report_id}>{report.title} — {formatDate(report.created_at)}</option>)}
        </select>
        <button onClick={refresh}>🔄 Refresh</button>
        <button className="primary" onClick={download} disabled={!manifest || loadedReportId !== reportId}>⇩ Download HTML</button>
      </div>

      {error && <div className="report-error">❌ {error}</div>}
      {!loading && reports.length === 0 && !error && <div className="card report-empty">No generated reports yet. Complete a document analysis to create one.</div>}

      {manifest && <>
        <div className="card report-context">
          <span className="report-status">COMPLETE</span>
          <div><b>{manifest.title}</b><small>{manifest.page_count} pages · {manifest.invocation_count} specialist invocations · {manifest.session_id} · {formatDate(manifest.created_at)}</small></div>
        </div>
        <div className="report-subtabs">
          {VIEW_TABS.map(([id, label]) => <button key={id} className={view === id ? 'tab-active' : ''} onClick={() => setView(id)}>{label}</button>)}
        </div>

        {view === 'overview' && <ReportOverview manifest={manifest} openPage={openPage} />}
        {view === 'reader' && currentPage && <PageReader
          manifest={manifest}
          page={currentPage}
          pageIndex={pageIndex}
          setPageIndex={setPageIndex}
          imageUrl={imageUrl}
          enhancedImageUrl={enhancedImageUrl}
          imageTab={imageTab}
          setImageTab={setImageTab}
          xml={xml}
          spineTab={spineTab}
          setSpineTab={setSpineTab}
          pageSpecialists={pageSpecialists}
          onIndex={() => setView('overview')}
        />}
        {view === 'audit' && <AuditTrail manifest={manifest} documentSpine={documentSpine} openPage={openPage} />}
      </>}
    </div>
  )
}

function ReportOverview({ manifest, openPage }) {
  return <>
    <div className="report-summary-grid">
      <div className="card report-summary"><span>ANALYSIS SUMMARY</span><h2>{manifest.title}</h2><p>{manifest.summary}</p><div className="report-chips"><i>{manifest.page_count} pages correlated</i><i>{manifest.specialist_count} unique specialists</i><i>{manifest.element_count} structured elements</i><i>Page spines retained</i></div></div>
      <div className="card report-metrics"><Metric label="Pages" value={manifest.page_count} /><Metric label="Invocations" value={manifest.invocation_count} /><Metric label="Complete" value={manifest.complete_count} /><Metric label="Failed" value={manifest.failed_count} /></div>
    </div>
    <div className="report-section-title">Document pages <span>Compare each analysis image with its canonical page spine</span></div>
    <div className="report-page-grid">{(manifest.pages || []).map((page, index) => <div className="card report-page-card" key={page.page_number}><div className="report-page-preview">📄<b>Page {page.page_number}</b></div><div className="report-page-meta"><div><b>Page {page.page_number}</b><em>COMPLETE</em></div><p>{page.summary}</p><div className="report-dots">{(page.specialists || []).map((_, dot) => <i key={dot} />)}</div><button onClick={() => openPage(index)}>Open page comparison</button></div></div>)}</div>
  </>
}

function Metric({ label, value }) {
  return <div><label>{label}</label><b>{value ?? 0}</b></div>
}

// The copy control sits outside the scrolling element so it stays pinned to the
// visible corner instead of scrolling away with the document.
function XmlPane({ xml }) {
  if (!xml) return <pre className="report-code">Loading XML...</pre>

  let body
  try {
    body = <ShikiHighlighter language="xml" theme="github-dark">{xml}</ShikiHighlighter>
  } catch {
    // Highlighting is cosmetic; unreadable XML is not worth losing the pane over.
    body = <pre>{xml}</pre>
  }

  return (
    <div className="report-xml-pane">
      <div className="report-xml-copy">
        <CopyButton getText={() => xml} label="Copy raw XML" />
      </div>
      <div className="report-code report-xml-scroll">{body}</div>
    </div>
  )
}

function PageReader({ manifest, page, pageIndex, setPageIndex, imageUrl, enhancedImageUrl, imageTab, setImageTab, xml, spineTab, setSpineTab, pageSpecialists, onIndex }) {
  const tabs = [['rendered', 'Rendered Spine'], ['xml', 'Raw XML'], ['results', 'Specialists'], ['audit', 'Audit']]
  return <>
    <div className="report-page-nav"><button onClick={onIndex}>⌂ Index</button><button disabled={pageIndex === 0} onClick={() => setPageIndex(pageIndex - 1)}>◀ Previous</button><b>Page {page.page_number} of {manifest.pages.length}</b><button disabled={pageIndex >= manifest.pages.length - 1} onClick={() => setPageIndex(pageIndex + 1)}>Next ▶</button></div>
    <div className="report-page-context">Spine: <code>{page.spine_key}</code> · {pageSpecialists.length} specialists · Keyboard ← → navigates pages</div>
    <div className="report-reader-grid">
      <div className="card report-image-pane">
        <div className="report-pane-head">🖼 Page Image <span>durable report copy</span></div>
        {/* Tabs only when there is a second image to switch to. The original is what
            the correlation artifact names as its source; the enhanced copy is what
            the image enhancer produced and what most specialists actually read. */}
        {enhancedImageUrl && <div className="report-spine-tabs">
          {[['original', 'Original'], ['enhanced', 'Enhanced']].map(([id, label]) =>
            <button key={id} className={imageTab === id ? 'active' : ''} onClick={() => setImageTab(id)}>{label}</button>)}
        </div>}
        <div className="report-image-canvas">
          {imageTab === 'enhanced' && enhancedImageUrl
            ? <img src={enhancedImageUrl} alt={`Enhanced page ${page.page_number}`} />
            : imageUrl
              ? <img src={imageUrl} alt={`Page ${page.page_number} as analysed`} />
              : <span>Loading image...</span>}
        </div>
      </div>
      <div className="card report-spine-pane"><div className="report-pane-head">🌳 Correlated Page Spine <span>schema v2.0</span></div><div className="report-spine-tabs">{tabs.map(([id, label]) => <button key={id} className={spineTab === id ? 'active' : ''} onClick={() => setSpineTab(id)}>{label}</button>)}</div>
        {spineTab === 'rendered' && <div className="report-spine-scroll"><div className="report-callout"><b>Page {page.page_number} synthesis</b><br />{page.summary}</div><div className="report-tree">{(page.elements || []).map((element, index) => <div className="report-tree-node" key={element.id || index} style={{ marginLeft: Math.min(element.depth || 0, 5) * 16 }}><span>{element.tag || 'P'}</span>{element.text}<small>{element.id}</small></div>)}</div></div>}
        {spineTab === 'xml' && <XmlPane xml={xml} />}
        {spineTab === 'results' && <div className="report-spine-scroll">{pageSpecialists.map((specialist, index) => <div className="report-result" key={`${specialist.name}-${index}`}><b>{specialist.name}</b><em>✓ COMPLETE</em><small>{specialist.s3_uri || 'Artifact retained'}</small></div>)}</div>}
        {spineTab === 'audit' && <div className="report-spine-scroll">{(page.audit || []).length ? page.audit.map((record, index) => <div className="report-result" key={`${record.specialist}-${index}`}><b>{record.specialist}</b><em className={record.status === 'FAILED' ? 'failed' : ''}>{record.status}</em><small>{formatDate(record.started_at)} → {formatDate(record.completed_at)} · {duration(record.started_at, record.completed_at)}</small></div>) : <p className="report-muted">No page timing records available.</p>}</div>}
      </div>
    </div>
  </>
}

function AuditTrail({ manifest, documentSpine, openPage }) {
  return <div className="report-audit-grid"><div><div className="card report-audit-card"><h3>Overall execution</h3><div className="report-audit-stats"><Metric label="Invocations" value={manifest.invocation_count} /><Metric label="Complete" value={manifest.complete_count} /><Metric label="Failed" value={manifest.failed_count} /><Metric label="Page Spines" value={manifest.page_count} /></div></div><div className="card report-audit-card"><h3>Per-page breakdown</h3>{(manifest.pages || []).map((page, index) => <div className="report-audit-row" key={page.page_number}><b>P{page.page_number}</b><span>{page.specialists?.length || 0} specialists · {page.elements?.length || 0} elements · canonical page spine retained</span><button onClick={() => openPage(index)}>View →</button></div>)}</div><div className="card report-audit-card"><h3>Document-wide correlated index</h3><pre className="report-code compact">{documentSpine}</pre></div></div><aside><div className="card report-audit-card"><h3>Run identity</h3><dl><dt>User</dt><dd>{manifest.user_name}</dd><dt>Session</dt><dd>{manifest.session_id}</dd><dt>Job</dt><dd>{manifest.job_id || manifest.report_id}</dd><dt>Document</dt><dd>{manifest.doc_id || '—'}</dd><dt>Generated</dt><dd>{formatDate(manifest.created_at)}</dd></dl></div><div className="card report-audit-card"><h3>Source</h3><p className="report-muted">{manifest.source_document_path}</p><h3>Specialists</h3><div className="report-chips">{(manifest.specialists || []).map(name => <i key={name}>{name}</i>)}</div></div></aside></div>
}
