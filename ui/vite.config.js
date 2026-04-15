import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

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
    plugins: [react(), configReloadPlugin()],
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
