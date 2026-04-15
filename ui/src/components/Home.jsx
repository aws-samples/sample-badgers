import React, { useState, useEffect } from 'react'

export default function Home({ onNavigate, branding = {} }) {
  const [env, setEnv] = useState(null)

  useEffect(() => {
    async function fetchEnv(retries = 10, delay = 500) {
      for (let i = 0; i < retries; i++) {
        try {
          const res = await fetch('/api/env')
          if (!res.ok) throw new Error(res.status)
          setEnv(await res.json())
          return
        } catch {
          await new Promise(r => setTimeout(r, delay))
        }
      }
    }
    fetchEnv()
  }, [])

  const name = branding.appName || 'BADGERS'
  const emoji = branding.appEmoji || '🦡'
  const description = branding.appDescription || ''

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
