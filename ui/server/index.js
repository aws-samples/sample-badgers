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

import { requireAuth } from './auth.js';
import { mountCoreRoutes } from './routes/core.js';
import { mountAdminRoutes } from './routes/admin.js';

const PROJECT_ROOT = resolve(__dirname, '../..');
const DIST_DIR = resolve(__dirname, '../dist');

const app = express();

// ── Proxy trust ───────────────────────────────────────────────────────────
// In production this server sits behind the load balancer that
// CfnExpressGatewayService provisions (deployment/stacks/ecs_stack.py), and an ALB
// sets X-Forwarded-For by default in append mode. Without this setting req.ip
// resolves to the balancer for every request, so the rate limiters below collapse
// into a single global bucket shared by all callers rather than one per user.
//
// 1 trusts exactly one hop, the balancer. `true` trusts the whole chain, which lets
// a client forge X-Forwarded-For to get its own bucket and which express-rate-limit
// rejects as ERR_ERL_PERMISSIVE_TRUST_PROXY. Raise this only if another proxy such
// as CloudFront is placed in front of the balancer.
app.set('trust proxy', 1);

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

// The one anonymous route: the load balancer's health check. Declared once here because
// three things below key off it — the rate-limit budget, the auth exemption, and the
// route itself in routes/core.js. Must stay in step with health_check_path in
// deployment/stacks/ecs_stack.py.
const HEALTHCHECK_PATH = '/api/healthcheck';

// ── Rate limiting ─────────────────────────────────────────────────────────
// Three separate budgets. A single shared one cannot serve all three callers: the
// balancer's health check must never be starved by user traffic, and a page load
// pulls far more static requests than API calls.
const RATE_LIMIT_DEFAULTS = {
    windowMs: 60 * 1000,
    standardHeaders: true,
    legacyHeaders: false,
};

const limiter = rateLimit({ ...RATE_LIMIT_DEFAULTS, max: 100 });

// /api/healthcheck is the balancer's health_check_path (ecs_stack.py) and the one
// unauthenticated /api route. It gets its own budget rather than sharing the 100/min
// above, because exhausting that budget would return 429 to the health check and get
// the task killed. The 3.1.0 changelog records this same failure from a blanket
// requireAuth mount returning 401 to health checks.
//
// A dedicated budget rather than a full exemption. The route is the only
// unauthenticated one under /api/, so it should not be the one path with no ceiling
// at all. The ceiling is far above any real health check interval.
const healthLimiter = rateLimit({ ...RATE_LIMIT_DEFAULTS, max: 600 });

app.use('/api/', (req, res, next) => {
    if (req.originalUrl.split('?')[0] === HEALTHCHECK_PATH) return healthLimiter(req, res, next);
    return limiter(req, res, next);
});

// Every /api/* route requires a verified Cognito bearer token except the health check,
// which must answer unauthenticated because the load balancer carries no token
// (health_check_path in deployment/stacks/ecs_stack.py). Mounted before the route groups
// so no other handler can be reached unauthenticated.
//
// This is the only anonymous route in the app, so it is deliberately the only entry in
// the set and the handler returns a constant. Branding used to be fetched here too; it
// is now bundled at build time by the badgers-branding plugin in ui/vite.config.js.
//
// originalUrl is used rather than req.path because Express strips the '/api' mount
// prefix from req.url inside path-mounted middleware.
const PUBLIC_API_PATHS = new Set([HEALTHCHECK_PATH]);
app.use('/api/', (req, res, next) => {
    if (PUBLIC_API_PATHS.has(req.originalUrl.split('?')[0])) return next();
    return requireAuth(req, res, next);
});

// No /api/me. The browser already holds a validated ID token from the OIDC flow, so it
// reads its own email, name and cognito:groups from those claims (ui/src/hooks/useUser.js)
// rather than asking the server to echo them back.
//
// The role that comes from those claims only drives which tabs are rendered. It is not an
// access-control decision: every admin route enforces requireAdmin server-side
// (ui/server/routes/admin.js), so a client claiming admin gains nothing.

// ── Mount route groups ──
mountCoreRoutes(app, PROJECT_ROOT);
mountAdminRoutes(app, PROJECT_ROOT);

// ── Static serving ──
const PORT = process.env.PORT || 7860;
const HOST = process.env.HOST || '127.0.0.1';

async function startServer() {
    await loadSSMConfig();

    if (process.env.NODE_ENV === 'production') {
        app.use(express.static(DIST_DIR));
        // Express 5 (path-to-regexp v8) rejects a bare '*' path; named wildcard
        // syntax is required. SPA fallback for client-side routes such as
        // /callback, which the OIDC redirect lands on.
        //
        // Rate limited because the handler reaches the file system (CodeQL
        // js/missing-rate-limiting, alert 42). The budget is generous relative to
        // /api/: this serves one small cached file and a user hits it on every
        // navigation and refresh, so a tight limit would throttle real browsing.
        const staticLimiter = rateLimit({ ...RATE_LIMIT_DEFAULTS, max: 600 });
        app.get('/*splat', staticLimiter, (req, res, next) => {
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

    app.listen(PORT, HOST, () => {
        console.log(`\n🦡 BADGERS Unified UI on http://${HOST}:${PORT}\n`);
    });
}

startServer();
