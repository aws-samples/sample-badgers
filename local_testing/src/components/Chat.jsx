import { useState, useEffect, useCallback, useMemo } from 'react'
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  useMessage,
  ThreadPrimitive,
  MessagePrimitive,
  ComposerPrimitive,
} from '@assistant-ui/react'
import Markdown from 'react-markdown'
import ShikiHighlighter from 'react-shiki'

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

function MarkdownContent({ text }) {
  if (!text) return null
  try {
    return (
      <Markdown components={{ code: CodeBlock }}>
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

// ── Messages ──

function UserMessage() {
  return (
    <MessagePrimitive.Root style={{ marginBottom: 8 }}>
      <div className="chat-message user">
        <MessagePrimitive.Parts>
          {({ part }) => {
            if (part.type === 'text') return <span>{part.text}</span>
            return null
          }}
        </MessagePrimitive.Parts>
      </div>
    </MessagePrimitive.Root>
  )
}

function AssistantMessage() {
  const message = useMessage()
  const isRunning = message.status?.type === 'running'
  const hasText = message.content?.some(c => c.type === 'text' && c.text?.trim())
  const reasoningParts = message.content?.filter(c => c.type === 'reasoning') || []

  if (!hasText && isRunning) {
    return (
      <MessagePrimitive.Root style={{ marginBottom: 8 }}>
        <div className="chat-message assistant">
          <LoadingDots />
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
      </div>
    </MessagePrimitive.Root>
  )
}

// ── Thread + Composer ──

function MyThread() {
  return (
    <ThreadPrimitive.Root style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <ThreadPrimitive.Viewport style={{ flex: 1, overflow: 'auto', padding: 12 }}>
        <ThreadPrimitive.Empty>
          <div style={{ color: 'var(--text-dim)', fontSize: 13, textAlign: 'center', marginTop: 200 }}>
            Send a message to start a conversation
          </div>
        </ThreadPrimitive.Empty>
        <ThreadPrimitive.Messages>
          {({ message }) => message.role === 'user' ? <UserMessage /> : <AssistantMessage />}
        </ThreadPrimitive.Messages>
      </ThreadPrimitive.Viewport>
      <MyComposer />
    </ThreadPrimitive.Root>
  )
}

function MyComposer() {
  return (
    <ComposerPrimitive.Root style={{
      display: 'flex', gap: 8,
      padding: '8px 12px', borderTop: '1px solid var(--border)',
    }}>
      <ComposerPrimitive.AddAttachment style={{
        background: 'none', border: '1px solid var(--border)',
        color: 'var(--text-dim)', padding: '6px 10px',
        borderRadius: 'var(--radius)', fontSize: 13, cursor: 'pointer',
      }}>
        📎
      </ComposerPrimitive.AddAttachment>
      <ComposerPrimitive.Input
        placeholder="Ask your agent something..."
        style={{
          flex: 1, background: 'var(--bg)', border: '1px solid var(--border)',
          color: 'var(--text)', padding: '8px 10px', borderRadius: 'var(--radius)',
          fontSize: 13, outline: 'none',
        }}
      />
      <ComposerPrimitive.Send style={{
        background: '#1f6feb', border: '1px solid #1f6feb', color: '#fff',
        padding: '6px 16px', borderRadius: 'var(--radius)', fontSize: 13, cursor: 'pointer',
      }}>
        Send
      </ComposerPrimitive.Send>
    </ComposerPrimitive.Root>
  )
}

// ── S3 Upload Attachment Adapter ──

class S3AttachmentAdapter {
  accept = 'application/pdf'

  async add({ file }) {
    if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
      throw new Error('Only PDF files are supported')
    }
    return {
      id: crypto.randomUUID(),
      type: 'document',
      name: file.name,
      contentType: 'application/pdf',
      file,
      status: { type: 'requires-action', reason: 'composer-send' },
    }
  }

  async send(attachment) {
    const formData = new FormData()
    formData.append('file', attachment.file)

    const res = await fetch('/api/upload', { method: 'POST', body: formData })
    const data = await res.json()

    if (data.error) throw new Error(data.error)

    // Return the s3 URI as text content so the agent sees it
    return {
      ...attachment,
      status: { type: 'complete' },
      content: [{
        type: 'text',
        text: `Uploaded file: ${data.s3Uri}`,
      }],
    }
  }

  async remove() {}
}

// ── Chat with adapter ──

function ChatInner() {
  const [tools, setTools] = useState(null)
  const [toolsLoading, setToolsLoading] = useState(false)
  const [toolsOpen, setToolsOpen] = useState(false)
  const [sessionId, setSessionId] = useState(genSessionId)
  const [auditMode, setAuditMode] = useState(false)
  const [dynamicTokens, setDynamicTokens] = useState(false)

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
      const messageText = lastUserMsg?.content
        ?.filter(c => c.type === 'text')
        .map(c => c.text)
        .join('\n') || ''

      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: messageText,
          session_id: sessionId,
          audit_mode: auditMode,
          dynamic_tokens: dynamicTokens,
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
          } catch {}
        }

        const content = []
        if (reasoning) content.push({ type: 'reasoning', text: reasoning })
        if (text.trim()) content.push({ type: 'text', text })
        if (content.length) yield { content }
      }
    },
  }), [sessionId, auditMode, dynamicTokens])

  const runtime = useLocalRuntime(adapter, {
    adapters: { attachments: attachmentAdapter },
  })

  const newSession = () => setSessionId(genSessionId())

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 260px', gap: 12, height: 600 }}>
      <div className="card" style={{ minHeight: 0, overflow: 'hidden' }}>
        <AssistantRuntimeProvider runtime={runtime}>
          <MyThread />
        </AssistantRuntimeProvider>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
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
          <button onClick={newSession} style={{ fontSize: 11, width: '100%' }}>🔄 New Session</button>
        </div>
      </div>
    </div>
  )
}

export default function Chat() {
  return <ChatInner />
}
