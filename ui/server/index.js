import express from 'express';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import { resolve, dirname } from 'path';
import { readFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { SSMClient, GetParametersCommand } from '@aws-sdk/client-ssm';

// Load .env before any modules that read process.env.
// In production the ECS task injects every value from SSM Parameter Store, so
// this file is a local-development convenience only.
const __dirname = dirname(fileURLToPath(import.meta.url));
const envPath = resolve(__dirname, '../config/.env');
if (existsSync(envPath)) {
    for (const line of readFileSync(envPath, 'utf-8').split('\n')) {
        const m = line.match(/^([A-Z0-9_]+)=(.*)$/);
        if (m && !(m[1] in process.env)) process.env[m[1]] = m[2].replace(/^["']|["']$/g, '');
    }
}

// ── Load any missing config from SSM Parameter Store ──────────────────────
// The ECS task already injects these as container secrets; this is a fallback
// for values added after a task definition was rendered, and for running the
// server outside ECS against a deployed environment. DEPLOYMENT_ID selects the
// parameter prefix.
const DEPLOYMENT_ID = process.env.DEPLOYMENT_ID || '';
const SSM_PARAM_SUFFIX_MAP = {
    'agentcore-runtime-websocket-arn': 'AGENTCORE_RUNTIME_WEBSOCKET_ARN',
    'agentcore-gateway-id': 'AGENTCORE_GATEWAY_ID',
    'jobs-table-name': 'JOBS_TABLE_NAME',
    'config-bucket-name': 'S3_CONFIG_BUCKET',
    'output-bucket-name': 'S3_OUTPUT_BUCKET',
    'source-bucket-name': 'S3_UPLOAD_BUCKET',
    'cognito-user-pool-id': 'COGNITO_USER_POOL_ID',
    'cognito-domain': 'COGNITO_DOMAIN',
    'cognito-ui-client-id': 'COGNITO_UI_CLIENT_ID',
    'ws-timeout-minutes': 'WS_TIMEOUT_MINUTES',
};

async function loadSSMConfig() {
    if (!DEPLOYMENT_ID) return; // no prefix to read from

    const missing = Object.entries(SSM_PARAM_SUFFIX_MAP)
        .filter(([, envKey]) => !process.env[envKey])
        .map(([suffix, envKey]) => [`/badgers-${DEPLOYMENT_ID}/${suffix}`, envKey]);

    if (missing.length === 0) return;

    const ssm = new SSMClient({ region: process.env.AWS_REGION || 'us-west-2' });
    const byName = Object.fromEntries(missing);

    try {
        const resp = await ssm.send(
            new GetParametersCommand({ Names: Object.keys(byName), WithDecryption: false })
        );
        for (const param of resp.Parameters || []) {
            const envKey = byName[param.Name];
            if (envKey && param.Value) {
                process.env[envKey] = param.Value;
                console.log(`[config] ${envKey} loaded from SSM`);
            }
        }
        if (resp.InvalidParameters?.length) {
            console.warn('[config] SSM params not found:', resp.InvalidParameters);
        }
    } catch (e) {
        console.warn('[config] SSM load failed (continuing with env vars):', e.message);
    }
}

import { getUser, requireAuth } from './auth.js';
import { mountCoreRoutes } from './routes/core.js';
import { mountAdminRoutes } from './routes/admin.js';

const PROJECT_ROOT = resolve(__dirname, '../..');
const DIST_DIR = resolve(__dirname, '../dist');

const app = express();

// ── CORS — same-origin only unless an explicit origin is configured ────────
// The React bundle is served from this same server in production, so no
// cross-origin access is needed by default.
const allowedOrigins = process.env.CORS_ALLOWED_ORIGIN
    ? [process.env.CORS_ALLOWED_ORIGIN]
    : [];
app.use(cors({ origin: allowedOrigins, credentials: true }));
app.use(express.json({ limit: '10mb' }));

// ── Request logging ───────────────────────────────────────────────────────
app.use((req, res, next) => {
    const start = Date.now();
    res.on('finish', () => {
        console.log(`${req.method} ${req.path} ${res.statusCode} ${Date.now() - start}ms`);
    });
    next();
});

const limiter = rateLimit({
    windowMs: 60 * 1000,
    max: 100,
    standardHeaders: true,
    legacyHeaders: false,
});
app.use('/api/', limiter);

// Every /api/* route requires a verified Cognito bearer token except /api/env, which
// must answer unauthenticated: it is the ALB health check path (health_check_path in
// deployment/stacks/ecs_stack.py), and the SPA reads it for branding before login.
// This matches the media-contracts reference implementation, which passes requireAuth
// into mountRoutes and applies it per route, leaving /api/env open. Mounted before the
// route groups so no other handler can be reached unauthenticated.
//
// originalUrl is used rather than req.path because Express strips the '/api' mount
// prefix from req.url inside path-mounted middleware.
const PUBLIC_API_PATHS = new Set(['/api/env']);
app.use('/api/', (req, res, next) => {
    if (PUBLIC_API_PATHS.has(req.originalUrl.split('?')[0])) return next();
    return requireAuth(req, res, next);
});

// ── User identity ──
app.get('/api/me', async (req, res) => {
    try {
        const user = req.user || (await getUser(req));
        if (!user) return res.status(401).json({ error: 'Authentication required' });
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
    await loadSSMConfig();

    if (process.env.NODE_ENV === 'production') {
        app.use(express.static(DIST_DIR));
        // Express 5 (path-to-regexp v8) rejects a bare '*' path; named wildcard
        // syntax is required. SPA fallback for client-side routes such as
        // /callback, which the OIDC redirect lands on.
        app.get('/*splat', (req, res, next) => {
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
