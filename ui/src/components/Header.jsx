import React from 'react'
import { useUser } from '../hooks/useUser.js'

const THEMES = [
    { id: 'dark', label: '🌑 Dark' },
    { id: 'purple', label: '🟣 AWS Purple' },
]

export default function Header({ branding = {}, theme = 'dark', onThemeChange }) {
    const { email, role, loading } = useUser()
    const name = branding.appName || ''
    const emoji = branding.appEmoji || '🦡'
    const logo = branding.appLogo || ''
    const logoHeight = branding.appLogoHeight || 32
    const subtitle = branding.appSubtitle || ''
    const description = branding.appDescription || ''

    return (
        <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
                <h1 style={{ fontSize: 22, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 10 }}>
                    {logo
                        ? <img src={logo} alt="" style={{ height: logoHeight }} />
                        : <span style={{ fontSize: 28 }}>{emoji}</span>
                    } {name}
                </h1>
                <p style={{ color: 'var(--text-dim)', fontSize: 13, marginTop: 4 }}>
                    {subtitle}
                </p>
                <p style={{ color: 'var(--text-dim)', fontSize: 13, marginTop: 4 }}>
                {description}
                </p>
            </div>
            {!loading && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-dim)' }}>
                    <select
                        value={theme}
                        onChange={e => onThemeChange?.(e.target.value)}
                        className="theme-toggle"
                        style={{ width: 'auto' }}
                    >
                        {THEMES.map(t => (
                            <option key={t.id} value={t.id}>{t.label}</option>
                        ))}
                    </select>
                    <span>{email}</span>
                    <span style={{
                        padding: '2px 8px',
                        borderRadius: 4,
                        fontSize: 11,
                        fontWeight: 600,
                        background: role === 'admin' ? 'var(--accent-subtle)' : 'var(--green-subtle)',
                        color: role === 'admin' ? 'var(--accent)' : 'var(--green)',
                        border: `1px solid ${role === 'admin' ? 'var(--accent-border)' : 'var(--green-border)'}`,
                    }}>
                        {role}
                    </span>
                </div>
            )}
        </div>
    )
}
