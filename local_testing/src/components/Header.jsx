import React from 'react'

export default function Header() {
  return (
    <div style={{ marginBottom: 24 }}>
      <h1 style={{ fontSize: 22, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 10 }}>
        <span style={{ fontSize: 28 }}>🦡</span> BADGERS Local Testing
      </h1>
      <p style={{ color: 'var(--text-dim)', fontSize: 13, marginTop: 4 }}>
        Test harness for document analysis agents
      </p>
    </div>
  )
}
