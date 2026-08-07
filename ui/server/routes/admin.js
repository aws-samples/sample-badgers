import { execFile, spawn } from 'child_process';
import { readFile, writeFile, readdir, stat } from 'fs/promises';
import { realpathSync } from 'fs';
import { resolve } from 'path';
import { requireAdmin } from '../auth.js';

export function mountAdminRoutes(app, PROJECT_ROOT) {
    const DEPLOY_DIR = resolve(PROJECT_ROOT, 'deployment');
    const APP_PY = resolve(DEPLOY_DIR, 'app.py');
    const S3_FILES_DIR = resolve(DEPLOY_DIR, 's3_files');
    const DEPLOY_CONFIG = resolve(DEPLOY_DIR, 'deployment_config.json');

    // Stack names are BADGERS-{Name}-{suffix}. The suffix is per-deployment, so it
    // cannot be baked in — the ECS stack injects STACK_SUFFIX into this container.
    // `id` is the PascalCase stack name; `key` is the stable id used by the client
    // and by the deploy/destroy routes.
    const STACK_PREFIX = 'BADGERS';
    const STACK_SUFFIX = process.env.STACK_SUFFIX || '';
    const STACKS = [
        { key: 's3', id: 'S3', name: 'S3 Buckets', description: 'Config + source + output buckets' },
        { key: 'cognito', id: 'Cognito', name: 'Cognito Auth', description: 'User pool with UI (OIDC/PKCE) and Gateway (M2M) clients' },
        { key: 'dynamodb', id: 'DynamoDB', name: 'DynamoDB Jobs', description: 'Jobs table for doc/job/subtask tracking' },
        { key: 'iam', id: 'IAM', name: 'IAM Roles', description: 'Lambda execution role with Bedrock/S3/DynamoDB permissions' },
        { key: 'ecr', id: 'ECR', name: 'ECR Registry', description: 'Container image registry' },
        { key: 'inference-profiles', id: 'InferenceProfiles', name: 'Inference Profiles', description: 'Cost tracking profiles per model' },
        { key: 'lambda', id: 'Lambda', name: 'Lambda Specialists', description: 'Serverless specialist functions + layers' },
        { key: 'xray', id: 'XRay', name: 'X-Ray Tracing', description: 'Transaction search for AgentCore tracing' },
        { key: 'gateway', id: 'Gateway', name: 'AgentCore Gateway', description: 'MCP Gateway with Lambda targets' },
        { key: 'memory', id: 'Memory', name: 'AgentCore Memory', description: 'Session persistence' },
        { key: 'runtime-websocket', id: 'RuntimeWebSocket', name: 'AgentCore Runtime', description: 'Strands agent with WebSocket streaming' },
        { key: 'vpc', id: 'Vpc', name: 'VPC', description: 'Private subnets and VPC endpoints for the UI service' },
        { key: 'ecs', id: 'ECS', name: 'UI (ECS)', description: 'Unified UI on an ECS Express Gateway service' },
        { key: 'custom-specialists', id: 'CustomSpecialists', name: 'Custom Specialists', description: 'Wizard-created specialists (optional)' },
    ];

    // BADGERS-{Name}-{suffix}
    const stackName = (id) => `${STACK_PREFIX}-${id}-${STACK_SUFFIX}`;

    const stackByKey = (key) => STACKS.find(s => s.key === key || s.id === key);

    // ── Helpers ──

    function execPromise(cmd, args) {
        return new Promise((resolve, reject) => {
            execFile(cmd, args, { maxBuffer: 10 * 1024 * 1024, timeout: 30000 }, (err, stdout) => {
                if (err) reject(err); else resolve(stdout);
            });
        });
    }

    function sseStream(res, cmd, args, extraEnv = {}) {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.flushHeaders();
        const resolvedArgs = args.map(a => a.endsWith('.sh') && !a.startsWith('/') && !a.startsWith('./') ? `./${a}` : a);
        const proc = spawn(cmd, resolvedArgs, {
            cwd: DEPLOY_DIR,
            // stdin is closed, so any interactive `read` in the script sees EOF.
            // Scripts must be driven non-interactively (see BADGERS_ASSUME_YES).
            stdio: ['ignore', 'pipe', 'pipe'],
            env: {
                ...process.env,
                PATH: [`${process.env.HOME}/.local/bin`, `${process.env.HOME}/.cargo/bin`, '/usr/local/bin', '/opt/homebrew/bin', process.env.PATH].join(':'),
                TERM: 'dumb',
                ...extraEnv,
            },
        });
        res.write(`data: ${JSON.stringify({ type: 'stdout', text: `▶ Running: ${cmd} ${resolvedArgs.join(' ')}\n` })}\n\n`);
        const heartbeat = setInterval(() => { try { res.write(`: heartbeat\n\n`); } catch { } }, 15000);
        const timeout = setTimeout(() => { if (!proc.killed) { proc.kill('SIGTERM'); setTimeout(() => { if (!proc.killed) proc.kill('SIGKILL'); }, 5000); } }, 45 * 60 * 1000);
        const cleanup = () => { clearInterval(heartbeat); clearTimeout(timeout); };
        proc.stdout.on('data', d => { try { res.write(`data: ${JSON.stringify({ type: 'stdout', text: d.toString() })}\n\n`); } catch { } });
        proc.stderr.on('data', d => { try { res.write(`data: ${JSON.stringify({ type: 'stderr', text: d.toString() })}\n\n`); } catch { } });
        proc.on('error', (err) => { cleanup(); try { res.write(`data: ${JSON.stringify({ type: 'stderr', text: `Process error: ${err.message}` })}\n\n`); res.write(`data: ${JSON.stringify({ type: 'done', code: 1 })}\n\n`); res.end(); } catch { } });
        proc.on('close', (code, signal) => { cleanup(); try { res.write(`data: ${JSON.stringify({ type: 'done', code: code ?? (signal ? 1 : 0) })}\n\n`); res.end(); } catch { } });
        res.on('close', () => { cleanup(); });
        return proc;
    }

    async function findJsonFiles(dir, rel) {
        const results = [];
        for (const entry of await readdir(dir)) {
            if (entry.startsWith('.')) continue;
            const full = resolve(dir, entry);
            const relPath = rel ? `${rel}/${entry}` : entry;
            if ((await stat(full)).isDirectory()) results.push(...await findJsonFiles(full, relPath));
            else if (entry.endsWith('.json')) results.push(relPath);
        }
        return results;
    }

    async function parseDeploymentTags() {
        const src = await readFile(APP_PY, 'utf-8');
        const match = src.match(/deployment_tags\s*=\s*\{([^}]+)\}/s);
        if (!match) return {};
        const tags = {};
        for (const line of match[1].split('\n')) { const m = line.match(/"(\w+)":\s*"([^"]*)"/); if (m) tags[m[1]] = m[2]; }
        return tags;
    }

    async function writeDeploymentTags(tags) {
        let src = await readFile(APP_PY, 'utf-8');
        const entries = Object.entries(tags).map(([k, v]) => `    "${k}": "${v}",`).join('\n');
        src = src.replace(/deployment_tags\s*=\s*\{[^}]+\}/s, `deployment_tags = {\n${entries}\n}`);
        await writeFile(APP_PY, src);
    }

    // ── All admin routes use requireAdmin middleware ──

    app.get('/api/config', requireAdmin, async (_req, res) => {
        try {
            const tags = await parseDeploymentTags();
            const src = await readFile(APP_PY, 'utf-8');
            const m = src.match(/CDK_DEFAULT_REGION.*?"(\S+?)"/);
            res.json({ tags, region: m ? m[1] : 'us-west-2' });
        } catch (e) { res.status(500).json({ error: e.message }); }
    });

    app.put('/api/config', requireAdmin, async (req, res) => {
        try { if (req.body.tags) await writeDeploymentTags(req.body.tags); res.json({ ok: true }); }
        catch (e) { res.status(500).json({ error: e.message }); }
    });

    app.get('/api/stacks', requireAdmin, async (_req, res) => {
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
                const name = stackName(s.id);
                const cf = cfStacks.find(c => c.StackName === name);
                return { ...s, stackName: name, status: cf ? cf.StackStatus : 'NOT_DEPLOYED', lastUpdated: cf ? cf.LastUpdatedTime || cf.CreationTime : null };
            }));
        } catch { res.json(STACKS.map(s => ({ ...s, stackName: stackName(s.id), status: 'UNKNOWN', lastUpdated: null }))); }
    });

    app.get('/api/stacks/:stackId/outputs', requireAdmin, async (req, res) => {
        const entry = stackByKey(req.params.stackId);
        if (!entry) return res.status(404).json({ error: `Unknown stack ${req.params.stackId}` });
        const name = stackName(entry.id);
        try {
            const result = await execPromise('aws', ['cloudformation', 'describe-stacks', '--stack-name', name, '--output', 'json']);
            const outputs = (JSON.parse(result).Stacks?.[0]?.Outputs || []).map(o => ({ key: o.OutputKey, value: o.OutputValue, description: o.Description || '' }));
            res.json(outputs);
        } catch { res.status(404).json({ error: `Stack ${name} not found` }); }
    });

    // Resolve a client-supplied stack key to its real, suffixed stack name.
    function resolveStackName(stackId) {
        if (!stackId) return null;
        const entry = stackByKey(stackId);
        return entry ? stackName(entry.id) : null;
    }

    app.post('/api/deploy', requireAdmin, (req, res) => {
        const { stackId } = req.body;
        const target = resolveStackName(stackId);
        if (stackId && !target) return res.status(400).json({ error: `Unknown stack ${stackId}` });
        const args = ['run', 'cdk', 'deploy', target || '--all', '--require-approval', 'never'];
        sseStream(res, 'uv', args);
    });

    app.post('/api/destroy', requireAdmin, (req, res) => {
        const target = resolveStackName(req.body.stackId);
        if (req.body.stackId && !target) {
            return res.status(400).json({ error: `Unknown stack ${req.body.stackId}` });
        }
        sseStream(res, 'uv', ['run', 'cdk', 'destroy', target || '--all', '--force']);
    });

    app.post('/api/sync-s3', requireAdmin, (_req, res) => { sseStream(res, 'bash', ['sync_s3_files.sh']); });

    // Full deploy runs the root deploy.sh in non-interactive mode (option 9).
    // BADGERS_ASSUME_YES skips its confirmation prompts, which would otherwise block
    // on a stream with no tty attached.
    const fullDeploy = (_req, res) => sseStream(
        res, 'bash', [resolve(PROJECT_ROOT, 'deploy.sh'), '9'],
        { BADGERS_ASSUME_YES: '1' },
    );
    app.post('/api/deploy-all', requireAdmin, fullDeploy);
    app.get('/api/deploy-all', requireAdmin, fullDeploy);

    app.get('/api/deployment-config', requireAdmin, async (_req, res) => {
        try { res.json(JSON.parse(await readFile(DEPLOY_CONFIG, 'utf-8'))); }
        catch (e) { res.status(500).json({ error: e.message }); }
    });

    app.put('/api/deployment-config', requireAdmin, async (req, res) => {
        try { await writeFile(DEPLOY_CONFIG, JSON.stringify(req.body, null, 4) + '\n'); res.json({ ok: true }); }
        catch (e) { res.status(500).json({ error: e.message }); }
    });

    app.get('/api/s3-configs', requireAdmin, async (_req, res) => {
        try { res.json(await findJsonFiles(S3_FILES_DIR, '')); }
        catch (e) { res.status(500).json({ error: e.message }); }
    });

    const S3_CONFIG_PREFIX = '/api/s3-configs/';
    app.use(S3_CONFIG_PREFIX, requireAdmin, async (req, res) => {
        const relPath = decodeURIComponent(req.path.replace(/^\//, ''));
        if (!relPath || relPath.includes('..')) return res.status(400).json({ error: 'Invalid path' });
        const fullPath = resolve(S3_FILES_DIR, relPath);
        // Normalize the path and verify it's within the allowed directory
        const normalizedPath = realpathSync(fullPath);
        if (!normalizedPath.startsWith(S3_FILES_DIR + '/')) return res.status(403).json({ error: 'Forbidden' });
        if (req.method === 'GET') {
            try { res.json({ path: relPath, content: JSON.parse(await readFile(fullPath, 'utf-8')) }); }
            catch { res.status(404).json({ error: `Not found: ${relPath}` }); }
        } else if (req.method === 'PUT') {
            try { await writeFile(fullPath, JSON.stringify(req.body.content, null, 4) + '\n'); res.json({ ok: true }); }
            catch (e) { res.status(500).json({ error: e.message }); }
        } else { res.status(405).json({ error: 'Method not allowed' }); }
    });
}
