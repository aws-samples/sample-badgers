import React, { useEffect, useState } from 'react'

export default function CopyButton({ getText, label = 'Copy' }) {
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!copied) return
    const timer = setTimeout(() => setCopied(false), 1500)
    return () => clearTimeout(timer)
  }, [copied])

  const copy = async () => {
    const text = getText()
    if (!text) return
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
    } catch {
      // navigator.clipboard is undefined on insecure origins and can be denied
      // by permission policy. Staying silent is better than a thrown error in
      // the render tree; the user sees no tick and can select the text.
    }
  }

  return (
    <button
      onClick={copy}
      title={copied ? 'Copied' : label}
      aria-label={label}
      style={{
        background: 'none', border: 'none', cursor: 'pointer', padding: '0 2px',
        fontSize: 12, lineHeight: 1,
        color: copied ? 'var(--green)' : 'var(--text-dim)',
      }}
    >
      {copied ? '✓' : '⧉'}
    </button>
  )
}
