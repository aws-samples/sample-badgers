import React, { useMemo, useEffect } from 'react'
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
  // Dynamically inject the theme stylesheet
  useEffect(() => {
    const id = `hljs-theme-${theme}`
    // Remove old theme link if present
    const old = document.getElementById('hljs-theme-link')
    if (old) old.remove()

    const link = document.createElement('link')
    link.id = 'hljs-theme-link'
    link.rel = 'stylesheet'
    link.href = `https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/${theme}.min.css`
    document.head.appendChild(link)

    return () => {
      const el = document.getElementById('hljs-theme-link')
      if (el) el.remove()
    }
  }, [theme])

  const html = useMemo(() => {
    try {
      return hljs.highlight(text, { language: 'json' }).value
    } catch {
      return text
    }
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
      dangerouslySetInnerHTML={{ __html: html + '\n' }}
    />
  )
}
