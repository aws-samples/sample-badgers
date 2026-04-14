import express from 'express';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import { resolve, dirname } from 'path';
import { readFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';

// Load .env before any modules that read process.env
const __dirname = dirname(fileURLToPath(import.meta.url));
const envPath = resolve(__dirname, '../config/.env');
if (existsSync(envPath)) {
    for (const line of readFileSync(envPath, 'utf-8').split('\n')) {
        const m = line.match(/^([A-Z0-9_]+)=(.*)$/);
        if (m && !(m[1] in process.env)) process.env[m[1]] = m[2].replace(/^["']|["']$/g, '');
    }
}

import { getUserFromOIDC, requireAuth } from './middleware/auth.js';
import { mountCoreRoutes } from './routes/core.js';
import { mountAdminRoutes } from './routes/admin.js';
const PROJECT_ROOT = resolve(__dirname, '../..');
const DIST_DIR = resolve(__dirname, '../dist');

const app = express();
const allowedOrigins = process.env.CORS_ALLOWED_ORIGIN
    ? [process.env.CORS_ALLOWED_ORIGIN]
    : [];
app.use(cors({ origin: allowedOrigins, credentials: true }));
app.use(express.json({ limit: '10mb' }));

const limiter = rateLimit({ windowMs: 60 * 1000, max: 100, standardHeaders: true, legacyHeaders: false });
app.use('/api/', limiter);
app.use('/api/', requireAuth);

// ── User identity ──

app.get('/api/me', async (req, res) => {
    try {
        const user = await getUserFromOIDC(req);
        res.json({
            email: user.email,
            name: user.name,
            role: user.groups.includes('admin') ? 'admin' : 'tester',
            verified: user.verified,
        });
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

// ── Mount route groups ──

mountCoreRoutes(app, PROJECT_ROOT);
mountAdminRoutes(app, PROJECT_ROOT);

// ── Static serving ──

const PORT = process.env.PORT || 7860;

async function startServer() {
    if (process.env.NODE_ENV === 'production') {
        app.use(express.static(DIST_DIR));
        app.get('*', (req, res, next) => {
            if (req.path.startsWith('/api')) return next();
            res.sendFile(resolve(DIST_DIR, 'index.html'));
        });
    } else {
        const { createServer: createViteServer } = await import('vite');
        const vite = await createViteServer({
            root: resolve(__dirname, '..'),
            server: { middlewareMode: true },
            appType: 'spa',
        });
        app.use(vite.middlewares);
    }

    app.listen(PORT, () => {
        console.log(`\n🦡 BADGERS Unified UI on http://localhost:${PORT}\n`);
    });
}

startServer();
