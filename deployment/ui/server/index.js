import express from 'express';
import cors from 'cors';
import { execFile, spawn } from 'child_process';
import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs';
import { readFile, writeFile, readdir, stat } from 'fs/promises';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DEPLOY_DIR = resolve(__dirname, '../..');
const APP_PY = resolve(DEPLOY_DIR, 'app.py');
const S3_FILES_DIR = resolve(DEPLOY_DIR, 's3_files');

const app = express();
app.use(cors());
app.use(express.json({ limit: '5mb' }));

const STACK_PREFIX = 'badgers';
const STACKS = [
    { id: 's3', name: 'S3 Buckets', description: 'Config + source + output buckets' },
    { id: 'cognito', name: 'Cognito Auth', description: 'OAuth 2.0 user pool & credentials' },
    { id: 'iam', name: 'IAM Roles', description: 'Lambda execution role with Bedrock/S3 permissions' },
    { id: 'ecr', name: 'ECR Registry', description: 'Container image registry' },
    { id: 'inference-profiles', name: 'Inference Profiles', description: 'Cost tracking profiles per model' },
    { id: 'lambda', name: 'Lambda Analyzers', description: 'Serverless analyzer functions + layers' },
    { id: 'xray', name: 'X-Ray Tracing', description: 'Transaction search for AgentCore tracing' },
    { id: 'gateway', name: 'AgentCore Gateway', description: 'MCP Gateway with Lambda targets' },
    { id: 'memory', name: 'AgentCore Memory', description: 'Session persistence' },
    { id: 'runtime-websocket', name: 'AgentCore Runtime', description: 'Strands agent with WebSocket streaming' },
];

// ── Helpers ──

function execPromise(cmd, args) {
    return new Promise((resolve, reject) => {
        execFile(cmd, args, { maxBuffer: 10 * 1024 * 1024, timeout: 30000 }, (err, stdout) => {
            if (err) reject(err); else resolve(stdout);
        });
    });
}

function sseStream(res, cmd, args) {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();
    // Prefix bare script names with ./ so they resolve relative to cwd
    const resolvedArgs = args.map(a =>
        a.endsWith('.sh') && !a.startsWith('/') && !a.startsWith('./') ? `./${a}` : a
    );
    console.log(`[SSE] Spawning: ${cmd} ${resolvedArgs.join(' ')} (cwd: ${DEPLOY_DIR})`);
    const proc = spawn(cmd, resolvedArgs, {
        cwd: DEPLOY_DIR,
        stdio: ['ignore', 'pipe', 'pipe'],
        env: {
            ...process.env,
            PATH: [
                `${process.env.HOME}/.local/bin`,
                `${process.env.HOME}/.cargo/bin`,
                '/usr/local/bin',
                '/opt/homebrew/bin',
                process.env.PATH,
            ].join(':'),
            TERM: 'dumb',
        },
    });
    res.write(`data: ${JSON.stringify({ type: 'stdout', text: `▶ Running: ${cmd} ${resolvedArgs.join(' ')}\n` })}\n\n`);
    // Heartbeat to keep connection alive during long deploys
    const heartbeat = setInterval(() => {
        try { res.write(`: heartbeat\n\n`); } catch { }
    }, 15000);
    // Kill process if it runs longer than 45 minutes
    const timeout = setTimeout(() => {
        if (!proc.killed) {
            proc.kill('SIGTERM');
            setTimeout(() => { if (!proc.killed) proc.kill('SIGKILL'); }, 5000);
        }
    }, 45 * 60 * 1000);
    const cleanup = () => { clearInterval(heartbeat); clearTimeout(timeout); };
    proc.stdout.on('data', d => {
        try { res.write(`data: ${JSON.stringify({ type: 'stdout', text: d.toString() })}\n\n`); } catch { }
    });
    proc.stderr.on('data', d => {
        try { res.write(`data: ${JSON.stringify({ type: 'stderr', text: d.toString() })}\n\n`); } catch { }
    });
    proc.on('error', (err) => {
        console.error(`[SSE] Process error: ${err.message}`);
        cleanup();
        try {
            res.write(`data: ${JSON.stringify({ type: 'stderr', text: `Process error: ${err.message}` })}\n\n`);
            res.write(`data: ${JSON.stringify({ type: 'done', code: 1 })}\n\n`);
            res.end();
        } catch { }
    });
    proc.on('close', (code, signal) => {
        console.log(`[SSE] Process closed with code: ${code}, signal: ${signal}`);
        cleanup();
        try {
            res.write(`data: ${JSON.stringify({ type: 'done', code: code ?? (signal ? 1 : 0) })}\n\n`);
            res.end();
        } catch { }
    });
    // Do NOT kill the child process on client disconnect — let deploys finish
    res.on('close', () => {
        console.log(`[SSE] Client disconnected (process still running: ${!proc.killed})`);
        cleanup();
    });
    return proc;
}

async function findJsonFiles(dir, rel) {
    const results = [];
    for (const entry of await readdir(dir)) {
        if (entry.startsWith('.')) continue;
        const full = resolve(dir, entry);
        const relPath = rel ? `${rel}/${entry}` : entry;
        if ((await stat(full)).isDirectory()) {
            results.push(...await findJsonFiles(full, relPath));
        } else if (entry.endsWith('.json')) {
            results.push(relPath);
        }
    }
    return results;
}

// ── Deployment Tags (app.py) ──

async function parseDeploymentTags() {
    const src = await readFile(APP_PY, 'utf-8');
    const match = src.match(/deployment_tags\s*=\s*\{([^}]+)\}/s);
    if (!match) return {};
    const tags = {};
    for (const line of match[1].split('\n')) {
        const m = line.match(/"(\w+)":\s*"([^"]*)"/);
        if (m) tags[m[1]] = m[2];
    }
    return tags;
}

async function writeDeploymentTags(tags) {
    let src = await readFile(APP_PY, 'utf-8');
    const entries = Object.entries(tags).map(([k, v]) => `    "${k}": "${v}",`).join('\n');
    src = src.replace(/deployment_tags\s*=\s*\{[^}]+\}/s, `deployment_tags = {\n${entries}\n}`);
    await writeFile(APP_PY, src);
}

app.get('/api/config', async (_req, res) => {
    try {
        const tags = await parseDeploymentTags();
        const src = await readFile(APP_PY, 'utf-8');
        const m = src.match(/CDK_DEFAULT_REGION.*?"(\S+?)"/);
        res.json({ tags, region: m ? m[1] : 'us-west-2' });
    } catch (e) { res.status(500).json({ error: e.message }); }
});

app.put('/api/config', async (req, res) => {
    try {
        if (req.body.tags) await writeDeploymentTags(req.body.tags);
        res.json({ ok: true });
    } catch (e) { res.status(500).json({ error: e.message }); }
});

// ── Stack Status ──

app.get('/api/stacks', async (_req, res) => {
    try {
        const result = await execPromise('aws', [
            'cloudformation', 'list-stacks', '--stack-status-filter',
            'CREATE_COMPLETE', 'UPDATE_COMPLETE', 'UPDATE_ROLLBACK_COMPLETE',
            'CREATE_IN_PROGRESS', 'UPDATE_IN_PROGRESS',
            'ROLLBACK_IN_PROGRESS', 'ROLLBACK_COMPLETE',
            'DELETE_IN_PROGRESS', 'CREATE_FAILED', 'DELETE_FAILED',
            '--output', 'json'
        ]);
        const cfStacks = JSON.parse(result).StackSummaries || [];
        res.json(STACKS.map(s => {
            const cf = cfStacks.find(c => c.StackName === `${STACK_PREFIX}-${s.id}`);
            return {
                ...s, stackName: `${STACK_PREFIX}-${s.id}`,
                status: cf ? cf.StackStatus : 'NOT_DEPLOYED',
                lastUpdated: cf ? cf.LastUpdatedTime || cf.CreationTime : null
            };
        }));
    } catch {
        res.json(STACKS.map(s => ({ ...s, stackName: `${STACK_PREFIX}-${s.id}`, status: 'UNKNOWN', lastUpdated: null })));
    }
});

app.get('/api/stacks/:stackId/outputs', async (req, res) => {
    const stackName = `${STACK_PREFIX}-${req.params.stackId}`;
    try {
        const result = await execPromise('aws', ['cloudformation', 'describe-stacks', '--stack-name', stackName, '--output', 'json']);
        const outputs = (JSON.parse(result).Stacks?.[0]?.Outputs || []).map(o => ({
            key: o.OutputKey, value: o.OutputValue, description: o.Description || ''
        }));
        res.json(outputs);
    } catch { res.status(404).json({ error: `Stack ${stackName} not found` }); }
});

// ── Deploy / Destroy / Sync (SSE streaming) ──

app.post('/api/deploy', (req, res) => {
    const { stackId, deploymentId } = req.body;
    const stackName = stackId ? `${STACK_PREFIX}-${stackId}` : '--all';
    const args = ['run', 'cdk', 'deploy', stackName, '--require-approval', 'never'];
    if (deploymentId) args.push('-c', `deployment_id=${deploymentId}`);
    sseStream(res, 'uv', args);
});

app.post('/api/destroy', (req, res) => {
    const stackName = req.body.stackId ? `${STACK_PREFIX}-${req.body.stackId}` : '--all';
    sseStream(res, 'uv', ['run', 'cdk', 'destroy', stackName, '--force']);
});

app.post('/api/sync-s3', (_req, res) => {
    sseStream(res, 'bash', ['sync_s3_files.sh']);
});

app.post('/api/deploy-all', (_req, res) => {
    sseStream(res, 'bash', ['deploy_from_scratch.sh', '--force']);
});

app.get('/api/deploy-all', (_req, res) => {
    sseStream(res, 'bash', ['deploy_from_scratch.sh', '--force']);
});

// Simulated deploy for frontend testing (no AWS calls)
app.get('/api/deploy-test', (_req, res) => {
    sseStream(res, 'bash', ['deploy_from_scratch.sh', '--force']);
});

// ── Deployment Config (analyzer selection) ──

const DEPLOY_CONFIG = resolve(DEPLOY_DIR, 'deployment_config.json');

app.get('/api/deployment-config', async (_req, res) => {
    try {
        res.json(JSON.parse(await readFile(DEPLOY_CONFIG, 'utf-8')));
    } catch (e) { res.status(500).json({ error: e.message }); }
});

app.put('/api/deployment-config', async (req, res) => {
    try {
        await writeFile(DEPLOY_CONFIG, JSON.stringify(req.body, null, 4) + '\n');
        res.json({ ok: true });
    } catch (e) { res.status(500).json({ error: e.message }); }
});

// ── S3 Config Files (JSON editor) ──

app.get('/api/s3-configs', async (_req, res) => {
    try { res.json(await findJsonFiles(S3_FILES_DIR, '')); }
    catch (e) { res.status(500).json({ error: e.message }); }
});

// Read/write use a middleware-style approach to handle deep paths (Express 5 compat)
const S3_CONFIG_PREFIX = '/api/s3-configs/';

app.use(S3_CONFIG_PREFIX, async (req, res) => {
    const relPath = decodeURIComponent(req.path.replace(/^\//, ''));
    if (!relPath || relPath.includes('..')) return res.status(400).json({ error: 'Invalid path' });
    const fullPath = resolve(S3_FILES_DIR, relPath);
    if (!fullPath.startsWith(S3_FILES_DIR)) return res.status(403).json({ error: 'Forbidden' });

    if (req.method === 'GET') {
        try {
            res.json({ path: relPath, content: JSON.parse(await readFile(fullPath, 'utf-8')) });
        } catch { res.status(404).json({ error: `Not found: ${relPath}` }); }
    } else if (req.method === 'PUT') {
        try {
            await writeFile(fullPath, JSON.stringify(req.body.content, null, 4) + '\n');
            res.json({ ok: true });
        } catch (e) { res.status(500).json({ error: e.message }); }
    } else {
        res.status(405).json({ error: 'Method not allowed' });
    }
});

// ── Start ──

const DIST_DIR = resolve(__dirname, '../dist');
const PORT = process.env.PORT || 3456;

async function startServer() {
    if (process.env.NODE_ENV === 'production') {
        app.use(express.static(DIST_DIR));
        app.get('*', (_req, res, next) => {
            if (_req.path.startsWith('/api')) return next();
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
        console.log(`BADGERS Deploy UI on http://localhost:${PORT}`);
        console.log(`CDK dir: ${DEPLOY_DIR}`);
    });
}

startServer();
