import express from 'express';
import cors from 'cors';
import { execFile, spawn } from 'child_process';
import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs';
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
        execFile(cmd, args, { maxBuffer: 10 * 1024 * 1024 }, (err, stdout) => {
            if (err) reject(err); else resolve(stdout);
        });
    });
}

function sseStream(res, cmd, args, opts) {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    const proc = spawn(cmd, args, { cwd: DEPLOY_DIR, shell: true, ...opts });
    proc.stdout.on('data', d => res.write(`data: ${JSON.stringify({ type: 'stdout', text: d.toString() })}\n\n`));
    proc.stderr.on('data', d => res.write(`data: ${JSON.stringify({ type: 'stderr', text: d.toString() })}\n\n`));
    proc.on('close', code => { res.write(`data: ${JSON.stringify({ type: 'done', code })}\n\n`); res.end(); });
    req_cleanup(res, proc);
    return proc;
}

function req_cleanup(res, proc) {
    res.req.on('close', () => { if (!proc.killed) proc.kill(); });
}

function findJsonFiles(dir, rel) {
    const results = [];
    for (const entry of readdirSync(dir)) {
        if (entry.startsWith('.')) continue;
        const full = resolve(dir, entry);
        const relPath = rel ? `${rel}/${entry}` : entry;
        if (statSync(full).isDirectory()) {
            results.push(...findJsonFiles(full, relPath));
        } else if (entry.endsWith('.json')) {
            results.push(relPath);
        }
    }
    return results;
}

// ── Deployment Tags (app.py) ──

function parseDeploymentTags() {
    const src = readFileSync(APP_PY, 'utf-8');
    const match = src.match(/deployment_tags\s*=\s*\{([^}]+)\}/s);
    if (!match) return {};
    const tags = {};
    for (const line of match[1].split('\n')) {
        const m = line.match(/"(\w+)":\s*"([^"]*)"/);
        if (m) tags[m[1]] = m[2];
    }
    return tags;
}

function writeDeploymentTags(tags) {
    let src = readFileSync(APP_PY, 'utf-8');
    const entries = Object.entries(tags).map(([k, v]) => `    "${k}": "${v}",`).join('\n');
    src = src.replace(/deployment_tags\s*=\s*\{[^}]+\}/s, `deployment_tags = {\n${entries}\n}`);
    writeFileSync(APP_PY, src);
}

app.get('/api/config', (_req, res) => {
    try {
        const tags = parseDeploymentTags();
        const src = readFileSync(APP_PY, 'utf-8');
        const m = src.match(/CDK_DEFAULT_REGION.*?"(\S+?)"/);
        res.json({ tags, region: m ? m[1] : 'us-west-2' });
    } catch (e) { res.status(500).json({ error: e.message }); }
});

app.put('/api/config', (req, res) => {
    try {
        if (req.body.tags) writeDeploymentTags(req.body.tags);
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
    const args = ['deploy', stackName, '--require-approval', 'never'];
    if (deploymentId) args.push('-c', `deployment_id=${deploymentId}`);
    sseStream(res, 'cdk', args);
});

app.post('/api/destroy', (req, res) => {
    const stackName = req.body.stackId ? `${STACK_PREFIX}-${req.body.stackId}` : '--all';
    sseStream(res, 'cdk', ['destroy', stackName, '--force']);
});

app.post('/api/sync-s3', (_req, res) => {
    sseStream(res, 'bash', ['sync_s3_files.sh']);
});

app.post('/api/deploy-all', (_req, res) => {
    sseStream(res, 'bash', ['deploy_from_scratch.sh']);
});

// ── Deployment Config (analyzer selection) ──

const DEPLOY_CONFIG = resolve(DEPLOY_DIR, 'deployment_config.json');

app.get('/api/deployment-config', (_req, res) => {
    try {
        res.json(JSON.parse(readFileSync(DEPLOY_CONFIG, 'utf-8')));
    } catch (e) { res.status(500).json({ error: e.message }); }
});

app.put('/api/deployment-config', (req, res) => {
    try {
        writeFileSync(DEPLOY_CONFIG, JSON.stringify(req.body, null, 4) + '\n');
        res.json({ ok: true });
    } catch (e) { res.status(500).json({ error: e.message }); }
});

// ── S3 Config Files (JSON editor) ──

app.get('/api/s3-configs', (_req, res) => {
    try { res.json(findJsonFiles(S3_FILES_DIR, '')); }
    catch (e) { res.status(500).json({ error: e.message }); }
});

// Read/write use a middleware-style approach to handle deep paths (Express 5 compat)
const S3_CONFIG_PREFIX = '/api/s3-configs/';

app.use(S3_CONFIG_PREFIX, (req, res) => {
    const relPath = decodeURIComponent(req.path.replace(/^\//, ''));
    if (!relPath || relPath.includes('..')) return res.status(400).json({ error: 'Invalid path' });
    const fullPath = resolve(S3_FILES_DIR, relPath);
    if (!fullPath.startsWith(S3_FILES_DIR)) return res.status(403).json({ error: 'Forbidden' });

    if (req.method === 'GET') {
        try {
            res.json({ path: relPath, content: JSON.parse(readFileSync(fullPath, 'utf-8')) });
        } catch { res.status(404).json({ error: `Not found: ${relPath}` }); }
    } else if (req.method === 'PUT') {
        try {
            writeFileSync(fullPath, JSON.stringify(req.body.content, null, 4) + '\n');
            res.json({ ok: true });
        } catch (e) { res.status(500).json({ error: e.message }); }
    } else {
        res.status(405).json({ error: 'Method not allowed' });
    }
});

// ── Start ──

const PORT = process.env.PORT || 3456;
app.listen(PORT, () => {
    console.log(`BADGERS Deploy API on http://localhost:${PORT}`);
    console.log(`CDK dir: ${DEPLOY_DIR}`);
});
