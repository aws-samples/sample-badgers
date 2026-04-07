import React from 'react'
import { useUser } from '../hooks/useUser.js'

export default function Header() {
    const { email, role, loading } = useUser()

    return (
        <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
                <h1 style={{ fontSize: 22, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span style={{ fontSize: 28 }}>🦡</span> BADGERS
                </h1>
                <p style={{ color: 'var(--text-dim)', fontSize: 13, marginTop: 4 }}>
                    Document analysis &amp; deployment console
                </p>
            </div>
            {!loading && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-dim)' }}>
                    <span>{email}</span>
                    <span style={{
                        padding: '2px 8px',
                        borderRadius: 4,
                        fontSize: 11,
                        fontWeight: 600,
                        background: role === 'admin' ? 'rgba(31, 111, 235, 0.15)' : 'rgba(63, 185, 80, 0.15)',
                        color: role === 'admin' ? '#58a6ff' : '#3fb950',
                        border: `1px solid ${role === 'admin' ? 'rgba(31, 111, 235, 0.3)' : 'rgba(63, 185, 80, 0.3)'}`,
                    }}>
                        {role}
                    </span>
                </div>
            )}
        </div>
    )
}
