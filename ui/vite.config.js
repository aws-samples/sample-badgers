import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],
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
