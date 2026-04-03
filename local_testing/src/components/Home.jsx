import React, { useState, useEffect } from 'react'

export default function Home({ onNavigate }) {
  const [env, setEnv] = useState(null)

  useEffect(() => {
    fetch('/api/env').then(r => r.json()).then(setEnv).catch(() => {})
  }, [])

  const tabs = [
    ['chat', '💬 Chat', 'Stream messages to the AgentCore Runtime via WebSocket'],
    ['editor', '✏️ Edit Analyzer', 'Modify analyzer prompts and configuration in S3'],
    ['wizard', '🧙 Create Analyzer', 'Build new analyzers with the guided wizard'],
    ['evaluator', '🧪 Evaluations', 'Review and score analyzer output quality'],
    ['pricing', '💰 Pricing', 'Estimate Bedrock costs for document workloads'],
    ['observability', '📊 Observability', 'View agent execution traces from CloudWatch'],
    ['chatlog', '📝 Chat Log', 'Browse historical chat sessions'],
  ]

  return (
    <div>
      <div className="card" style={{ marginBottom: 16 }}>
        <h2 style={{ fontSize: 18, marginBottom: 8 }}>🦡 BADGERS Test Interface</h2>
        <p style={{ color: 'var(--text-dim)', fontSize: 13 }}>
          Test harness for BADGERS (Broad Agentic Document Generative Extraction &amp; Recognition System),
          a vision-enabled AI system that processes documents using specialized analyzers.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 8, marginBottom: 16 }}>
        {tabs.map(([key, name, desc]) => (
          <div key={key} className="card home-card" style={{ padding: 12, cursor: 'pointer' }}
            onClick={() => onNavigate?.(key)}>
            <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 4 }}>{name}</div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>{desc}</div>
          </div>
        ))}
      </div>

      {env && (
        <div className="card" style={{ fontSize: 12 }}>
          <div style={{ fontWeight: 500, marginBottom: 8 }}>Environment</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 16px' }}>
            <span style={{ color: 'var(--text-dim)' }}>Region</span>
            <span style={{ fontFamily: 'SF Mono, Menlo, monospace' }}>{env.region || 'not set'}</span>
            <span style={{ color: 'var(--text-dim)' }}>Runtime ARN</span>
            <span style={{ fontFamily: 'SF Mono, Menlo, monospace', wordBreak: 'break-all' }}>{env.runtimeArn ? '✅ configured' : '❌ not set'}</span>
            <span style={{ color: 'var(--text-dim)' }}>Gateway ID</span>
            <span style={{ fontFamily: 'SF Mono, Menlo, monospace' }}>{env.gatewayId || '❌ not set'}</span>
            <span style={{ color: 'var(--text-dim)' }}>Config Bucket</span>
            <span style={{ fontFamily: 'SF Mono, Menlo, monospace' }}>{env.configBucket || '❌ not set'}</span>
          </div>
        </div>
      )}
    </div>
  )
}
