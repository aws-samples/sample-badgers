import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'
import { existsSync, readFileSync } from 'fs'

/**
 * Branding is build-time config, not an API.
 *
 * config/branding.json is read here and exposed as an importable module, so App.jsx
 * gets it as a static import. If the file is absent the defaults below are used, so a
 * fresh clone still builds — the file is gitignored (ui/.gitignore) and intentionally
 * per-developer. Whatever is present at build time is bundled; nothing is read at
 * runtime.
 *
 * Previously this came from GET /api/env, which re-read the file synchronously on
 * every request. That endpoint is the load balancer's health check path and has to be
 * unauthenticated, so it meant continuous blocking file I/O on the one route reachable
 * without a token, to deliver a value that cannot change after the image is built.
 */
const BRANDING_ID = 'virtual:badgers-branding'
const BRANDING_RESOLVED = '\0' + BRANDING_ID
const BRANDING_DEFAULTS = {
    appName: 'BADGERS',
    appEmoji: '🦡',
    appSubtitle: 'Document analysis & deployment console',
    appDescription: '',
    appLogo: '',
    appLogoHeight: 32,
    theme: 'dark',
}

function brandingPlugin() {
    const brandingPath = resolve(__dirname, 'config/branding.json')
    const read = () => {
        try {
            if (existsSync(brandingPath)) {
                return { ...BRANDING_DEFAULTS, ...JSON.parse(readFileSync(brandingPath, 'utf-8')) }
            }
        } catch (e) {
            console.warn(`[branding] ignoring unreadable ${brandingPath}: ${e.message}`)
        }
        return BRANDING_DEFAULTS
    }
    return {
        name: 'badgers-branding',
        resolveId(id) {
            if (id === BRANDING_ID) return BRANDING_RESOLVED
        },
        load(id) {
            if (id === BRANDING_RESOLVED) return `export default ${JSON.stringify(read())}`
        },
        configureServer(server) {
            // Invalidate the virtual module so an edit to branding.json is picked up by
            // the full-reload that configReloadPlugin already fires.
            server.watcher.add(brandingPath)
            server.watcher.on('change', (file) => {
                if (file !== brandingPath) return
                const mod = server.moduleGraph.getModuleById(BRANDING_RESOLVED)
                if (mod) server.moduleGraph.invalidateModule(mod)
            })
        },
    }
}

/** Watch config/*.json and trigger full page reload on change */
function configReloadPlugin() {
    return {
        name: 'config-reload',
        configureServer(server) {
            const configDir = resolve(__dirname, 'config')
            server.watcher.add(configDir)
            server.watcher.on('change', (file) => {
                if (file.startsWith(configDir) && file.endsWith('.json')) {
                    server.ws.send({ type: 'full-reload' })
                }
            })
        },
    }
}

export default defineConfig({
    plugins: [react(), configReloadPlugin(), brandingPlugin()],
    server: {
        port: 5175,
        proxy: {
            '/api': {
                target: 'http://localhost:7860',
                configure: (proxy) => {
                    proxy.on('proxyRes', (proxyRes, _req, res) => {
                        if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
                            res.setHeader('Content-Type', 'text/event-stream');
                            res.setHeader('Cache-Control', 'no-cache');
                            res.setHeader('Connection', 'keep-alive');
                            res.setHeader('X-Accel-Buffering', 'no');
                            res.flushHeaders();
                        }
                    });
                },
                timeout: 3600000,
                proxyTimeout: 3600000,
            }
        }
    }
})
