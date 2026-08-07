import React from 'react'

// The "Environment" panel that listed the region, gateway id and bucket names is gone.
// It was display-only — nothing here used those values, since the browser never calls
// S3, DynamoDB or AgentCore directly — so it served deployment identifiers to a browser
// for no functional reason. Deployment state is visible in the CloudFormation console
// and in deploy.sh output.
export default function Home({ onNavigate, branding = {} }) {
  const name = branding.appName || 'BADGERS'
  const emoji = branding.appEmoji || '🦡'
  const description = branding.appDescription || ''

  const tabs = [
    ['chat', '💬 Chat', 'Stream messages to the AgentCore Runtime via WebSocket'],
    ['editor', '✏️ Edit Specialist', 'Modify specialist prompts and configuration in S3'],
    ['wizard', '🧙 Create Specialist', 'Build new specialists with the guided wizard'],
    ['evaluator', '🧪 Evaluations', 'Review and score specialist output quality'],
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

    </div>
  )
}
