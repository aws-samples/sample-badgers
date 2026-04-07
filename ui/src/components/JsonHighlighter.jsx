import React, { useEffect, useRef } from 'react'
import hljs from 'highlight.js/lib/core'
import json from 'highlight.js/lib/languages/json'

hljs.registerLanguage('json', json)

export const THEMES = [
  'github-dark',
  'github-dark-dimmed',
  'atom-one-dark',
  'monokai',
  'dracula',
  'nord',
  'tokyo-night-dark',
  'vs2015',
  'an-old-hope',
  'androidstudio',
  'agate',
  'a11y-dark',
  'felipec',
  'ir-black',
  'nnfx-dark',
  'obsidian',
  'paraiso-dark',
  'rainbow',
  'stackoverflow-dark',
  'sunburst',
  'xt256',
]

export default function JsonHighlighter({ text, theme = 'github-dark' }) {
  const codeRef = useRef(null)

  // Dynamically inject the theme stylesheet
  useEffect(() => {
    // Validate theme against allowlist to prevent injection via href
    const safeTheme = THEMES.includes(theme) ? theme : 'github-dark'

    // Remove old theme link if present
    const old = document.getElementById('hljs-theme-link')
    if (old) old.remove()

    const link = document.createElement('link')
    link.id = 'hljs-theme-link'
    link.rel = 'stylesheet'
    link.href = `https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/${safeTheme}.min.css`
    document.head.appendChild(link)

    return () => {
      const el = document.getElementById('hljs-theme-link')
      if (el) el.remove()
    }
  }, [theme])

  // Set text content and highlight via DOM API (no innerHTML)
  useEffect(() => {
    const el = codeRef.current
    if (!el) return
    el.textContent = text + '\n'
    el.removeAttribute('data-highlighted')
    hljs.highlightElement(el)
  }, [text])

  return (
    <pre
      className="hljs"
      style={{
        margin: 0, padding: 12,
        fontFamily: 'SF Mono, Menlo, monospace', fontSize: 13, lineHeight: 1.6,
        whiteSpace: 'pre-wrap', wordBreak: 'break-all',
        background: 'transparent',
        pointerEvents: 'none',
      }}
    >
      <code ref={codeRef} className="language-json" />
    </pre>
  )
}
