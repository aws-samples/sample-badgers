import { useState, useEffect, useCallback, useMemo } from 'react'
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  useAuiState,
  ThreadPrimitive,
  MessagePrimitive,
  ComposerPrimitive,
  AttachmentPrimitive,
} from '@assistant-ui/react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import ShikiHighlighter from 'react-shiki'
import CopyButton from './CopyButton'

function genSessionId() {
  return 'ws-session-' + crypto.randomUUID()
}

// ── Markdown with syntax highlighting ──

function CodeBlock({ className, children }) {
  const match = /language-(\w+)/.exec(className || '')
  const lang = match ? match[1] : 'text'
  const code = String(children).replace(/\n$/, '')

  if (!match || !code) {
    return <code className="inline-code">{children}</code>
  }

  try {
    return (
      <div className="code-block-wrapper">
        <div className="code-block-header">
          <span>{lang}</span>
        </div>
        <ShikiHighlighter language={lang} theme="github-dark">{code}</ShikiHighlighter>
      </div>
    )
  } catch {
    return (
      <pre className="code-block-wrapper" style={{ padding: 12 }}>
        <code>{code}</code>
      </pre>
    )
  }
}

// react-markdown only implements CommonMark, where tables, strikethrough and
// autolinks do not exist. Without remark-gfm the agent's pipe tables parsed as a
// single paragraph and CommonMark collapsed their newlines into spaces, which is
// why they rendered as one run-together line of pipes. Module scope, not inline,
// so the array identity is stable across renders while a response streams.
const REMARK_PLUGINS = [remarkGfm]

function MarkdownContent({ text }) {
  if (!text) return null
  try {
    return (
      <Markdown remarkPlugins={REMARK_PLUGINS} components={{ code: CodeBlock }}>
        {text}
      </Markdown>
    )
  } catch {
    return <span>{text}</span>
  }
}

// ── Chain of Thought ──

function Reasoning({ text }) {
  return <p className="reasoning-text">{text}</p>
}

function ChainOfThought({ parts }) {
  const [open, setOpen] = useState(false)
  if (!parts.length) return null
  return (
    <div className="thinking-block">
      <button className="thinking-trigger" onClick={() => setOpen(o => !o)}>
        <span>{open ? '▼' : '▶'} 🧠 Thinking</span>
      </button>
      {open && (
        <div className="thinking-content">
          {parts.map((p, i) => <Reasoning key={i} text={p.text} />)}
        </div>
      )}
    </div>
  )
}

function LoadingDots() {
  return <div className="loading-dots"><span /><span /><span /></div>
}

// ── Copy ──
// CopyButton lives in ./CopyButton so the Reports Raw XML pane can share it.

// ── Message text ──

// The composer keeps typed text in `content` and each attachment's parts in a
// separate `attachments` array; it never merges them. Both halves are collected
// here so the rendered bubble shows exactly what run() sends to the agent --
// otherwise an attachment-only message renders as an empty bubble.
function collectUserText(message) {
  const typed = (message?.content || [])
    .filter(c => c.type === 'text')
    .map(c => c.text)
  const attached = (message?.attachments || [])
    .flatMap(a => (a.content || []).filter(c => c.type === 'text').map(c => c.text))
  return [...typed, ...attached].join('\n\n')
}

function assistantCopyText(message) {
  const parts = message?.content || []
  const text = parts.filter(c => c.type === 'text').map(c => c.text).join('\n')
  const reasoning = parts.filter(c => c.type === 'reasoning').map(c => c.text).join('\n')
  if (!reasoning) return text
  return `${text}\n\n--- Thinking ---\n\n${reasoning}`
}

// ── Messages ──

function UserMessage() {
  const message = useAuiState((s) => s.message)
  const text = collectUserText(message)

  return (
    <MessagePrimitive.Root style={{ marginBottom: 8 }}>
      <div className="chat-message user" style={{ display: 'flex', alignItems: 'flex-start', gap: 6 }}>
        <span style={{ flex: 1, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{text}</span>
        <CopyButton getText={() => text} label="Copy message" />
      </div>
    </MessagePrimitive.Root>
  )
}

function AssistantMessage({ activity }) {
  const message = useAuiState((s) => s.message)
  const isRunning = message.status?.type === 'running'
  const hasText = message.content?.some(c => c.type === 'text' && c.text?.trim())
  const reasoningParts = message.content?.filter(c => c.type === 'reasoning') || []

  if (!hasText && isRunning) {
    return (
      <MessagePrimitive.Root style={{ marginBottom: 8 }}>
        <div className="chat-message assistant">
          <LoadingDots />
          {activity && (
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 4 }}>{activity}</div>
          )}
        </div>
      </MessagePrimitive.Root>
    )
  }

  return (
    <MessagePrimitive.Root style={{ marginBottom: 8 }}>
      <div className="chat-message assistant markdown-body">
        <MessagePrimitive.Parts>
          {({ part }) => {
            if (part.type === 'text' && part.text) return <MarkdownContent text={part.text} />
            return null
          }}
        </MessagePrimitive.Parts>
        <ChainOfThought parts={reasoningParts} />
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 2 }}>
          {/* Read at click time, not render time, so a copy during streaming
              takes whatever has arrived so far. */}
          <CopyButton getText={() => assistantCopyText(message)} label="Copy response and thinking" />
        </div>
      </div>
    </MessagePrimitive.Root>
  )
}

// ── Thread + Composer ──

function MyThread({ attachmentAdapter, activity }) {
  return (
    <ThreadPrimitive.Root style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <ThreadPrimitive.Viewport style={{ flex: 1, overflow: 'auto', padding: 12 }}>
        <ThreadPrimitive.Empty>
          <div style={{ color: 'var(--text-dim)', fontSize: 13, textAlign: 'center', marginTop: 200 }}>
            Send a message to start a conversation
          </div>
        </ThreadPrimitive.Empty>
        <ThreadPrimitive.Messages>
          {({ message }) => message.role === 'user' ? <UserMessage /> : <AssistantMessage activity={activity} />}
        </ThreadPrimitive.Messages>
        {/* Streamed reasoning is collapsed inside the message, so without this a
            long tool call looks like a hung UI. `activity` is the server's own
            status event: Connecting, Thinking, or Using <tool>. */}
        {activity && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            fontSize: 12, color: 'var(--accent)', padding: '4px 2px',
          }}>
            <span className="activity-spinner" aria-hidden="true" />
            <span>{activity}</span>
          </div>
        )}
      </ThreadPrimitive.Viewport>
      <MyComposer attachmentAdapter={attachmentAdapter} />
    </ThreadPrimitive.Root>
  )
}

// 'removed' has no entry on purpose: dropping an attachment clears the line.
const UPLOAD_STATUS_TEXT = {
  attached: e => `📎 ${e.filename} attached. It uploads when you press Send.`,
  uploading: e => `⏳ Uploading ${e.filename}...`,
  uploaded: e => `✓ Uploaded ${e.filename}`,
  error: e => `❌ ${e.filename}: ${e.message}`,
}

// Hoisted so ComposerPrimitive.Attachments can memoize on it. Rendered inside
// the attachment provider, which is what makes AttachmentPrimitive work here.
const renderAttachmentChip = ({ attachment }) => (
  <div style={{
    display: 'inline-flex', alignItems: 'center', gap: 6,
    background: 'var(--surface)', border: '1px solid var(--border)',
    borderRadius: 'var(--radius)', padding: '3px 6px 3px 8px',
    fontSize: 12, color: 'var(--text-dim)', maxWidth: 260,
  }}>
    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
      📄 <AttachmentPrimitive.Name />
    </span>
    <AttachmentPrimitive.Remove
      aria-label={`Remove ${attachment.name}`}
      title="Remove attachment"
      style={{
        background: 'none', border: 'none', color: 'var(--text-dim)',
        cursor: 'pointer', fontSize: 13, lineHeight: 1, padding: '0 2px',
      }}
    >
      ✕
    </AttachmentPrimitive.Remove>
  </div>
)

function MyComposer({ attachmentAdapter }) {
  const [upload, setUpload] = useState(null)

  useEffect(() => {
    if (!attachmentAdapter) return
    return attachmentAdapter.subscribe(setUpload)
  }, [attachmentAdapter])

  const statusText = upload && UPLOAD_STATUS_TEXT[upload.state]?.(upload)

  return (
    <div style={{ borderTop: '1px solid var(--border)' }}>
      {statusText && (
        <div style={{
          fontSize: 12, padding: '6px 12px 0',
          color: upload.state === 'error' ? 'var(--red)'
            : upload.state === 'uploaded' ? 'var(--green)'
              : 'var(--text-dim)',
        }}>
          {statusText}
        </div>
      )}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, padding: '6px 12px 0' }}>
        <ComposerPrimitive.Attachments>
          {renderAttachmentChip}
        </ComposerPrimitive.Attachments>
      </div>
      <ComposerPrimitive.Root style={{
        display: 'flex', gap: 8,
        padding: '8px 12px',
      }}>
        <ComposerPrimitive.AddAttachment style={{
          background: 'none', border: '1px solid var(--border)',
          color: 'var(--text-dim)', padding: '6px 10px',
          borderRadius: 'var(--radius)', fontSize: 13, cursor: 'pointer',
        }}>
          📎
        </ComposerPrimitive.AddAttachment>
        <ComposerPrimitive.Input
          placeholder="Provide an S3 URI or attach a PDF by clicking the paperclip icon."
          style={{
            flex: 1, background: 'var(--bg)', border: '1px solid var(--border)',
            color: 'var(--text)', padding: '8px 10px', borderRadius: 'var(--radius)',
            fontSize: 13, outline: 'none',
          }}
        />
        <ComposerPrimitive.Send style={{
          background: 'var(--accent-bg)', border: '1px solid var(--accent-bg)', color: '#fff',
          padding: '6px 16px', borderRadius: 'var(--radius)', fontSize: 13, cursor: 'pointer',
        }}>
          Send
        </ComposerPrimitive.Send>
      </ComposerPrimitive.Root>
    </div>
  )
}

// ── S3 Upload Attachment Adapter ──

class S3AttachmentAdapter {
  accept = 'application/pdf,image/png,image/jpeg,image/tiff,image/webp,image/gif'

  // Top level of the job-tracking hierarchy (doc_id -> job_id -> subtask_id).
  // The server mints it per upload; we hold the most recent one so subsequent
  // turns can attribute their jobs to the document being discussed.
  lastDocId = ''

  // The adapter is a plain class outside the React tree, so it cannot render
  // status itself. Components inside AssistantRuntimeProvider subscribe and do
  // it on its behalf.
  listeners = new Set()

  subscribe(listener) {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  notify(event) {
    for (const listener of this.listeners) listener(event)
  }

  // Holds the file in memory only. Nothing is uploaded here, so an attachment
  // the user removes -- or never sends -- leaves no object in S3.
  async add({ file }) {
    // Routing by type is deterministic: a PDF is rasterized into page images,
    // an image is already a single page. Accept both. The server re-checks by
    // magic bytes -- this client gate is just for a fast, friendly rejection.
    const name = file.name.toLowerCase()
    const okExt = /\.(pdf|png|jpe?g|tiff?|webp|gif)$/.test(name)
    const okMime = file.type === 'application/pdf' || file.type.startsWith('image/')
    if (!okExt && !okMime) {
      const message = 'Only PDF or image files (PNG, JPEG, TIFF, WebP, GIF) are supported'
      this.notify({ state: 'error', filename: file.name, message })
      throw new Error(message)
    }

    this.notify({ state: 'attached', filename: file.name })

    return {
      id: crypto.randomUUID(),
      type: 'document',
      name: file.name,
      contentType: file.type || 'application/octet-stream',
      file,
      status: { type: 'requires-action', reason: 'composer-send' },
    }
  }

  // Called by the composer only when the message is actually sent, which is
  // where the upload happens. The returned text part is what the agent reads
  // as its instruction; see the run() adapter, which pulls it out of
  // message.attachments.
  async send(attachment) {
    this.notify({ state: 'uploading', filename: attachment.name })

    try {
      const formData = new FormData()
      formData.append('file', attachment.file)

      const res = await fetch('/api/upload', { method: 'POST', body: formData })
      // The route reports failures as JSON, but a proxy or crash upstream can
      // return something else -- fall back to the status code instead of
      // throwing an unhelpful parse error.
      const data = await res.json().catch(() => ({}))
      if (!res.ok || data.error) throw new Error(data.error || `Upload failed (HTTP ${res.status})`)
      if (!data.s3Uri) throw new Error('Upload succeeded but the server returned no S3 URI')

      if (data.docId) this.lastDocId = data.docId

      this.notify({ state: 'uploaded', filename: attachment.name, s3Uri: data.s3Uri })

      return {
        ...attachment,
        status: { type: 'complete' },
        content: [{
          type: 'text',
          text: `Process: ${data.s3Uri}`,
        }],
      }
    } catch (e) {
      this.notify({ state: 'error', filename: attachment.name, message: e.message })
      throw e
    }
  }

  // Nothing was uploaded, so discarding an attachment is purely local.
  async remove(attachment) {
    this.notify({ state: 'removed', filename: attachment?.name })
  }
}

// ── Analyzer status ──

// The stream reports which tool is starting but never reports one finishing or
// failing, so terminal state has to come from the job records the specialist
// Lambdas write. /api/jobs/:jobId is the only source of real COMPLETE/FAILED.
const PILL_STYLE = {
  COMPLETE: { background: 'var(--green)', color: '#fff', borderColor: 'var(--green)' },
  FAILED: { background: 'var(--red)', color: '#fff', borderColor: 'var(--red)' },
  RUNNING: { background: 'var(--accent-bg)', color: '#fff', borderColor: 'var(--accent-bg)' },
  PENDING: { background: 'var(--surface)', color: 'var(--text-dim)', borderColor: 'var(--border)' },
}

// A specialist runs once per page, so one name can hold several subtasks. The
// pill reports the state that needs attention rather than the most common one.
const STATUS_RANK = { FAILED: 0, RUNNING: 1, PENDING: 2, COMPLETE: 3 }

function groupSpecialists(subtasks) {
  const byName = new Map()
  for (const task of subtasks || []) {
    const name = task.specialist || String(task.subtask_id || '').split('#')[0] || 'unknown'
    const entry = byName.get(name) || { name, status: task.status, error: '', total: 0, complete: 0 }
    entry.total += 1
    if (task.status === 'COMPLETE') entry.complete += 1
    if ((STATUS_RANK[task.status] ?? 9) < (STATUS_RANK[entry.status] ?? 9)) entry.status = task.status
    if (task.status === 'FAILED' && task.error && !entry.error) entry.error = task.error
    byName.set(name, entry)
  }
  return [...byName.values()].sort((a, b) => a.name.localeCompare(b.name))
}

function useJobStatus(jobId, active) {
  const [job, setJob] = useState(null)

  useEffect(() => {
    if (!jobId) { setJob(null); return }
    let cancelled = false
    let timer

    const poll = async () => {
      try {
        const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`)
        // 404 until the first specialist writes its row, 503 when the deployment
        // has no jobs table. Both mean "nothing to show", not an error to raise.
        if (res.ok) {
          const data = await res.json()
          if (!cancelled) setJob(data)
        }
      } catch { /* transient; the next tick retries */ }
      // Re-polls only while the turn runs. When `active` goes false the effect
      // re-runs and this fires once more, letting the pills settle on their
      // terminal state. 2.5s keeps well inside the server's 100 req/min budget.
      if (!cancelled && active) timer = setTimeout(poll, 2500)
    }

    poll()
    return () => { cancelled = true; clearTimeout(timer) }
  }, [jobId, active])

  return job
}

// RUNNING gets a spinner element instead of a glyph. Colour comes from
// currentColor so one rule works on every pill background.
const PILL_GLYPH = {
  COMPLETE: '✓',
  FAILED: '✕',
  PENDING: '·',
}

function AnalyzerPills({ job }) {
  const specialists = useMemo(() => groupSpecialists(job?.subtasks), [job])
  if (!specialists.length) return null

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
      {specialists.map(s => {
        const pages = s.total > 1 ? ` (${s.complete}/${s.total} pages)` : ''
        // Carried on aria-label as well as title: a title alone is not reliably
        // announced, and the glyph itself is decorative.
        const label = `${s.name.replace(/_/g, ' ')}: ${s.status}${pages}${s.error ? ` — ${s.error}` : ''}`
        return (
          <span
            key={s.name}
            title={label}
            aria-label={label}
            style={{
              ...(PILL_STYLE[s.status] || PILL_STYLE.PENDING),
              display: 'inline-flex', alignItems: 'center', gap: 4,
              border: '1px solid', borderRadius: 999,
              padding: '2px 8px', fontSize: 10, whiteSpace: 'nowrap',
            }}
          >
            {s.status === 'RUNNING'
              ? <span className="pill-spinner" aria-hidden="true" />
              : <span aria-hidden="true" style={{ fontSize: 11, lineHeight: 1 }}>
                  {PILL_GLYPH[s.status] || PILL_GLYPH.PENDING}
                </span>}
            <span>
              {s.name.replace(/_/g, ' ')}
              {s.total > 1 && ` ${s.complete}/${s.total}`}
            </span>
          </span>
        )
      })}
    </div>
  )
}

// ── Chat with adapter ──

function ChatInner({ onNewSession }) {
  const [tools, setTools] = useState(null)
  const [toolsLoading, setToolsLoading] = useState(false)
  const [toolsOpen, setToolsOpen] = useState(false)
  // No setter: a new session remounts this component (see Chat below) rather than
  // swapping the id in place, so the id is fixed for the lifetime of the instance.
  const [sessionId] = useState(genSessionId)
  const [auditMode, setAuditMode] = useState(false)
  const [dynamicTokens, setDynamicTokens] = useState(false)
  // Non-null only while a turn is in flight; doubles as the "is running" flag.
  const [activity, setActivity] = useState(null)
  const [jobId, setJobId] = useState('')
  const job = useJobStatus(jobId, activity !== null)

  const refreshTools = useCallback(async () => {
    setToolsLoading(true)
    try {
      const res = await fetch('/api/tools')
      const data = await res.json()
      setTools(data.tools || [])
    } catch { setTools([]) }
    setToolsLoading(false)
  }, [])

  useEffect(() => { refreshTools() }, [refreshTools])

  const attachmentAdapter = useMemo(() => new S3AttachmentAdapter(), [])

  const adapter = useMemo(() => ({
    async *run({ messages, abortSignal }) {
      const lastUserMsg = [...messages].reverse().find(m => m.role === 'user')
      // Includes the attachment's `Process: <s3 uri>` part, which lives outside
      // `content`. Appending it means an empty composer sends just the
      // instruction, while typed text keeps it.
      const messageText = collectUserText(lastUserMsg)

      setActivity('Connecting...')
      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: messageText,
            session_id: sessionId,
            audit_mode: auditMode,
            dynamic_tokens: dynamicTokens,
            // Empty until a PDF has been uploaded in this session; the agent
            // treats an absent doc_id as "not attributable to a document".
            doc_id: attachmentAdapter.lastDocId,
          }),
          signal: abortSignal,
        })

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let text = ''
        let reasoning = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop()

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            try {
              const evt = JSON.parse(line.slice(6))
              if (evt.type === 'text') text += evt.text
              else if (evt.type === 'thinking') reasoning += evt.text
              else if (evt.type === 'error') text = `❌ ${evt.text}`
              // Previously discarded. 'status' is the only progress signal the
              // server sends, and 'job' is what drives the analyzer pills.
              else if (evt.type === 'status') setActivity(evt.text)
              else if (evt.type === 'job' && evt.job_id) setJobId(evt.job_id)
              else if (evt.type === 'done') setActivity(null)
            } catch {}
          }

          const content = []
          if (reasoning) content.push({ type: 'reasoning', text: reasoning })
          if (text.trim()) content.push({ type: 'text', text })
          if (content.length) yield { content }
        }
      } finally {
        // Covers abort and thrown errors too, so the indicator can't stick on.
        setActivity(null)
      }
    },
  }), [sessionId, auditMode, dynamicTokens, attachmentAdapter])

  const runtime = useLocalRuntime(adapter, {
    adapters: { attachments: attachmentAdapter },
  })

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 260px', gap: 12, height: 600 }}>
      <div className="card" style={{ minHeight: 0, overflow: 'hidden' }}>
        <AssistantRuntimeProvider runtime={runtime}>
          <MyThread attachmentAdapter={attachmentAdapter} activity={activity} />
        </AssistantRuntimeProvider>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, minHeight: 0, overflow: 'auto' }}>
        {job?.subtasks?.length > 0 && (
          <div className="card" style={{ padding: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <span style={{ fontSize: 12, fontWeight: 500 }}>🧩 Analyzers</span>
              <span style={{ fontSize: 10, color: 'var(--text-dim)' }}>{job.status}</span>
            </div>
            <AnalyzerPills job={job} />
          </div>
        )}
        <div className="card" style={{ padding: 12 }}>
          <label style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
            <input type="checkbox" checked={auditMode} onChange={e => setAuditMode(e.target.checked)} />
            🔍 Audit Mode
          </label>
          <label style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', marginTop: 6 }}>
            <input type="checkbox" checked={dynamicTokens} onChange={e => setDynamicTokens(e.target.checked)} />
            ⚡ Dynamic Tokens
          </label>
        </div>
        <div className="card" style={{ padding: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={{ fontSize: 12, fontWeight: 500, cursor: 'pointer' }} onClick={() => setToolsOpen(p => !p)}>
              {toolsOpen ? '▼' : '▶'} 🔧 Tools {tools ? `(${tools.length})` : ''}
            </span>
            <button onClick={refreshTools} disabled={toolsLoading} style={{ fontSize: 11, padding: '2px 8px' }}>↻</button>
          </div>
          {toolsOpen && (
            <div style={{ fontSize: 11, color: 'var(--text-dim)', maxHeight: 200, overflow: 'auto' }}>
              {toolsLoading ? 'Loading...' : tools?.map(t => <div key={t}>{t}</div>) || 'No tools'}
            </div>
          )}
        </div>
        <div className="card" style={{ padding: 12, fontSize: 12 }}>
          <div style={{ fontWeight: 500, marginBottom: 4 }}>Session</div>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', wordBreak: 'break-all', marginBottom: 8 }}>{sessionId}</div>
          {/* Deliberately enabled mid-run: abandoning a stalled turn is a reason to
              start a new session, not something to prevent. */}
          <button onClick={onNewSession} style={{ fontSize: 11, width: '100%' }}>🔄 New Session</button>
        </div>
      </div>
    </div>
  )
}

export default function Chat() {
  // "New Session" remounts ChatInner instead of resetting state piecemeal. A session
  // reset has to clear the thread, the composer's attachments and status line, the
  // activity indicator, the analyzer pills, and S3AttachmentAdapter.lastDocId --
  // and useLocalRuntime owns the message store, so no single call clears all of it.
  // Keying the subtree is the only reset that cannot leave a fragment behind.
  //
  // lastDocId is the one that matters most: it survives on the adapter instance, so
  // carrying it into a new session would attribute the next job to the previous
  // document, which is exactly the doc_id mismatch that breaks report generation.
  const [sessionInstance, setSessionInstance] = useState(0)

  return (
    <ChatInner
      key={sessionInstance}
      onNewSession={() => setSessionInstance(instance => instance + 1)}
    />
  )
}
