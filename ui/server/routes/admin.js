import { execFile, spawn } from 'child_process';
import { readFile, writeFile, readdir, stat } from 'fs/promises';
import { realpathSync } from 'fs';
import { resolve } from 'path';
import { requireAdmin } from '../auth.js';

export function mountAdminRoutes(app, PROJECT_ROOT) {
    const DEPLOY_DIR = resolve(PROJECT_ROOT, 'deployment');
    const S3_FILES_DIR = resolve(DEPLOY_DIR, 's3_files');
    const DEPLOY_CONFIG = resolve(DEPLOY_DIR, 'deployment_config.json');

    // Stack names carry the whole deployment identity:
    // BADGERS-{Name}-{deployment-id}-{suffix}. ECS supplies DEPLOYMENT_ID as that
    // composite resource identity; local development supplies both parts separately.
    // `id` is the PascalCase stack name; `key` is the stable id used by the client
    // and by the deploy/destroy routes.
    const STACK_PREFIX = 'BADGERS';
    const STACK_SUFFIX = process.env.STACK_SUFFIX || '';
    const DEPLOYMENT_ID = process.env.DEPLOYMENT_ID || '';
    const RESOURCE_ID = !DEPLOYMENT_ID || !STACK_SUFFIX || DEPLOYMENT_ID.endsWith(`-${STACK_SUFFIX}`)
        ? DEPLOYMENT_ID
        : `${DEPLOYMENT_ID}-${STACK_SUFFIX}`;
    const DEPLOYMENT_LABEL = STACK_SUFFIX && DEPLOYMENT_ID.endsWith(`-${STACK_SUFFIX}`)
        ? DEPLOYMENT_ID.slice(0, -(STACK_SUFFIX.length + 1))
        : DEPLOYMENT_ID;
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
        { key: 'observability', id: 'Observability', name: 'Observability', description: 'Optional CloudWatch dashboard, alarms, and notification handoff' },
    ];

    const stackName = (id) => `${STACK_PREFIX}-${id}-${RESOURCE_ID}`;

    const stackByKey = (key) => STACKS.find(s => s.key === key || s.id === key);

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
        const resolvedArgs = args.map(a => a.endsWith('.sh') && !a.startsWith('/') && !a.startsWith('./') ? `./${a}` : a);
        const proc = spawn(cmd, resolvedArgs, {
            cwd: DEPLOY_DIR,
            stdio: ['ignore', 'pipe', 'pipe'],
            env: {
                ...process.env,
                // sync_s3_files.sh expects the human deployment label and suffix
                // separately, while ECS passes this service the composite ID.
                DEPLOYMENT_ID: DEPLOYMENT_LABEL,
            },
        });
        res.write(`data: ${JSON.stringify({ type: 'stdout', text: `▶ Running: ${cmd} ${resolvedArgs.join(' ')}\n` })}\n\n`);
        const heartbeat = setInterval(() => { try { res.write(': heartbeat\n\n'); } catch { } }, 15000);
        const timeout = setTimeout(() => { if (!proc.killed) proc.kill('SIGTERM'); }, 45 * 60 * 1000);
        const cleanup = () => { clearInterval(heartbeat); clearTimeout(timeout); };
        proc.stdout.on('data', d => { try { res.write(`data: ${JSON.stringify({ type: 'stdout', text: d.toString() })}\n\n`); } catch { } });
        proc.stderr.on('data', d => { try { res.write(`data: ${JSON.stringify({ type: 'stderr', text: d.toString() })}\n\n`); } catch { } });
        proc.on('error', (err) => { cleanup(); try { res.write(`data: ${JSON.stringify({ type: 'stderr', text: `Process error: ${err.message}` })}\n\n`); res.write(`data: ${JSON.stringify({ type: 'done', code: 1 })}\n\n`); res.end(); } catch { } });
        proc.on('close', (code, signal) => { cleanup(); try { res.write(`data: ${JSON.stringify({ type: 'done', code: code ?? (signal ? 1 : 0) })}\n\n`); res.end(); } catch { } });
        res.on('close', cleanup);
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

    // ── All admin routes use requireAdmin middleware ──

    app.get('/api/config', requireAdmin, (_req, res) => {
        res.json({
            configuration: 'local-planner',
            command: './deploy.sh configure --ui',
            message: 'Deployment tags are configured per deployment ID in the local planner.',
        });
    });

    app.put('/api/config', requireAdmin, (_req, res) => {
        res.status(410).json({
            error: 'Deployment tags are configured with ./deploy.sh configure --ui; this application does not edit deployment source files.',
        });
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

    app.post('/api/sync-s3', requireAdmin, (_req, res) => {
        sseStream(res, 'bash', ['sync_s3_files.sh']);
    });

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
