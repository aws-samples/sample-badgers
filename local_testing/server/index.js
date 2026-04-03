import express from 'express';
import cors from 'cors';
import { readFileSync, readdirSync, existsSync, mkdirSync, appendFileSync, writeFileSync } from 'fs';
import { readFile, readdir, appendFile, writeFile } from 'fs/promises';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import WebSocket from 'ws';
import { SignatureV4 } from '@smithy/signature-v4';
import { HttpRequest } from '@smithy/protocol-http';
import { Sha256 } from '@aws-crypto/sha256-js';
import { fromNodeProviderChain } from '@aws-sdk/credential-providers';
import { BedrockAgentCoreControlClient, ListGatewayTargetsCommand } from '@aws-sdk/client-bedrock-agentcore-control';
import { S3Client, PutObjectCommand, GetObjectCommand, ListObjectsV2Command } from '@aws-sdk/client-s3';
import { CloudWatchLogsClient, StartQueryCommand, GetQueryResultsCommand } from '@aws-sdk/client-cloudwatch-logs';
import multer from 'multer';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(__dirname, '../..');
const DEPLOY_DIR = resolve(PROJECT_ROOT, 'deployment');
const LOCAL_TESTING_DIR = resolve(__dirname, '..');
const LOGS_DIR = resolve(LOCAL_TESTING_DIR, 'logs', 'chat_sessions');

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// ── Helpers ──

function loadEnvFile() {
    const envPath = resolve(LOCAL_TESTING_DIR, 'config', '.env');
    if (!existsSync(envPath)) return {};
    const env = {};
    for (const line of readFileSync(envPath, 'utf-8').split('\n')) {
        const m = line.match(/^([A-Z0-9_]+)=(.*)$/);
        if (m) env[m[1]] = m[2].replace(/^["']|["']$/g, '');
    }
    return env;
}

const ENV = loadEnvFile();
const REGION = ENV.AWS_REGION || process.env.AWS_REGION || 'us-west-2';
const RUNTIME_ARN = ENV.AGENTCORE_RUNTIME_WEBSOCKET_ARN || '';
const GATEWAY_ID = ENV.AGENTCORE_GATEWAY_ID || '';
const AWS_PROFILE = ENV.AWS_PROFILE || process.env.AWS_PROFILE || undefined;

const credentials = fromNodeProviderChain({ profile: AWS_PROFILE });

const controlClient = new BedrockAgentCoreControlClient({
    region: REGION,
    credentials,
});

const UPLOAD_BUCKET = ENV.S3_UPLOAD_BUCKET || '';
const OUTPUT_BUCKET = ENV.S3_OUTPUT_BUCKET || '';
const CONFIG_BUCKET = ENV.S3_CONFIG_BUCKET || '';
const WS_TIMEOUT_MIN = parseInt(ENV.WS_TIMEOUT_MINUTES, 10) || 30;
const s3Client = new S3Client({ region: REGION, credentials });
const cwLogsClient = new CloudWatchLogsClient({ region: REGION, credentials });
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });

console.log(`Region: ${REGION}`);
console.log(`Profile: ${AWS_PROFILE || 'default'}`);
console.log(`Runtime ARN: ${RUNTIME_ARN ? '✅' : '❌ not set'}`);
console.log(`Gateway ID: ${GATEWAY_ID || '❌ not set'}`);
console.log(`Upload Bucket: ${UPLOAD_BUCKET || '❌ not set'}`);

// ── SigV4 Presigned WebSocket URL ──

async function getPresignedWsUrl(sessionId) {
    // Build the WebSocket URL
    const host = `bedrock-agentcore.${REGION}.amazonaws.com`;
    const encodedArn = encodeURIComponent(RUNTIME_ARN);
    const path = `/runtimes/${encodedArn}/ws`;

    // Create the HTTP request to sign
    const request = new HttpRequest({
        method: 'GET',
        protocol: 'wss:',
        hostname: host,
        path,
        query: { 'X-Amzn-Bedrock-AgentCore-Runtime-Session-Id': sessionId },
        headers: { host },
    });

    const signer = new SignatureV4({
        credentials,
        region: REGION,
        service: 'bedrock-agentcore',
        sha256: Sha256,
    });

    const signed = await signer.presign(request, { expiresIn: 300 });

    // Build the full URL from signed request
    const qs = Object.entries(signed.query)
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
        .join('&');

    return `wss://${host}${path}?${qs}`;
}

// ── Environment Info ──

app.get('/api/env', (_req, res) => {
    res.json({
        region: REGION,
        runtimeArn: RUNTIME_ARN ? 'configured' : '',
        gatewayId: GATEWAY_ID,
        configBucket: ENV.S3_CONFIG_BUCKET || '',
        outputBucket: ENV.S3_OUTPUT_BUCKET || '',
    });
});

// ── Tools ──

app.get('/api/tools', async (_req, res) => {
    if (!GATEWAY_ID) return res.json({ tools: [], error: 'AGENTCORE_GATEWAY_ID not set' });
    try {
        const allTargets = [];
        let nextToken;
        do {
            const cmd = new ListGatewayTargetsCommand({
                gatewayIdentifier: GATEWAY_ID,
                ...(nextToken && { nextToken }),
            });
            const result = await controlClient.send(cmd);
            allTargets.push(...(result.items || []));
            nextToken = result.nextToken;
        } while (nextToken);

        const tools = allTargets.map(t => t.name).sort();
        console.log(`[tools] found ${tools.length} targets`);
        res.json({ tools });
    } catch (e) {
        console.error('[tools] error:', e.message);
        res.json({ tools: [], error: e.message });
    }
});

// ── File Upload (S3) ──

app.post('/api/upload', upload.single('file'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'No file provided' });
    if (!UPLOAD_BUCKET) return res.status(500).json({ error: 'S3_UPLOAD_BUCKET not configured' });

    // Validate PDF: check MIME type and magic bytes (%PDF- header)
    const isPdfMime = req.file.mimetype === 'application/pdf';
    const isPdfMagic = req.file.buffer.length >= 5 && req.file.buffer.slice(0, 5).toString() === '%PDF-';
    if (!isPdfMime || !isPdfMagic) {
        return res.status(400).json({ error: 'Only PDF files are accepted' });
    }

    const filename = req.file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_');
    const s3Key = `uploads/${Date.now()}_${filename}`;

    try {
        await s3Client.send(new PutObjectCommand({
            Bucket: UPLOAD_BUCKET,
            Key: s3Key,
            Body: req.file.buffer,
            ContentType: req.file.mimetype,
        }));
        const s3Uri = `s3://${UPLOAD_BUCKET}/${s3Key}`;
        console.log(`[upload] ${filename} → ${s3Uri}`);
        res.json({ s3Uri, filename, size: req.file.size });
    } catch (e) {
        console.error('[upload] error:', e.message);
        res.status(500).json({ error: e.message });
    }
});

// ── Chat (native WebSocket to AgentCore Runtime, SSE to frontend) ──

app.post('/api/chat', async (req, res) => {
    const { message, session_id, audit_mode, dynamic_tokens } = req.body;

    // ── Session logging ──
    mkdirSync(LOGS_DIR, { recursive: true });
    const logFile = resolve(LOGS_DIR, `${session_id}.log`);
    const log = (line) => { appendFile(logFile, line + '\n').catch(() => { }); };
    log(`\n${'='.repeat(60)}`);
    log(`[${new Date().toISOString()}] USER: ${message}`);
    log(`audit_mode=${audit_mode} dynamic_tokens=${dynamic_tokens}`);
    log('-'.repeat(60));

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const send = (type, text) => {
        res.write(`data: ${JSON.stringify({ type, text })}\n\n`);
        if (type === 'text' || type === 'thinking' || type === 'error') log(`[${type}] ${text}`);
    };

    if (!RUNTIME_ARN) {
        send('error', 'AGENTCORE_RUNTIME_WEBSOCKET_ARN not configured');
        res.end();
        return;
    }

    try {
        send('status', 'Connecting...');
        const wsUrl = await getPresignedWsUrl(session_id);
        console.log(`[chat] connecting to WebSocket for session ${session_id}`);

        const ws = new WebSocket(wsUrl);

        // Kill WebSocket if no completion after configured timeout
        const wsTimeout = setTimeout(() => {
            if (ws.readyState === WebSocket.OPEN) {
                console.log('[chat] WebSocket timeout — closing');
                send('error', `Request timed out after ${WS_TIMEOUT_MIN} minutes`);
                ws.close();
            }
        }, WS_TIMEOUT_MIN * 60 * 1000);

        let ended = false;
        const finish = () => {
            if (ended) return;
            ended = true;
            clearTimeout(wsTimeout);
            send('done', '');
            res.end();
        };

        ws.on('open', () => {
            console.log('[chat] WebSocket connected');
            send('status', 'Thinking...');
            ws.send(JSON.stringify({
                prompt: message,
                session_id,
                actor_id: 'local_testing_user',
                audit_mode: audit_mode || false,
                dynamic_tokens_enabled: dynamic_tokens || false,
            }));
        });

        ws.on('message', (raw) => {
            try {
                const str = raw.toString();
                const data = str.startsWith('data: ')
                    ? JSON.parse(str.slice(6))
                    : JSON.parse(str);

                // Extract reasoning/text from streaming events
                if (data.event && typeof data.event === 'object') {
                    const delta = data.event?.contentBlockDelta?.delta;
                    if (delta?.reasoningContent?.text) {
                        send('thinking', delta.reasoningContent.text);
                        return;
                    }
                    if (delta?.text) {
                        send('text', delta.text);
                        return;
                    }
                    // Other event types (messageStart, contentBlockStop, etc.) — skip
                    return;
                }

                // Skip duplicate reasoningText events (already captured via contentBlockDelta)
                if (data.reasoningText) return;

                // Skip duplicate data text events
                if ('data' in data && typeof data.data === 'string') return;

                // Status events
                if (data.current_tool_use) {
                    send('status', `Using ${data.current_tool_use.name || 'tool'}`);
                    log(`[tool] ${data.current_tool_use.name || 'unknown'}`);
                } else if (data.init_event_loop || data.start_event_loop) {
                    send('status', 'Thinking...');
                }

                // Completion
                if (data.complete || data.force_stop || data.type === 'error' || (data.result != null)) {
                    if (data.type === 'error') send('error', data.message || 'Unknown error');
                    log(`[done] ${new Date().toISOString()}`);
                    finish();
                    ws.close();
                }
            } catch (e) {
                console.log('[chat] parse error:', e.message, raw.toString().substring(0, 100));
            }
        });

        ws.on('error', (e) => {
            console.error('[chat] ws error:', e.message);
            send('error', e.message);
            finish();
        });

        ws.on('close', () => {
            console.log('[chat] WebSocket closed');
            finish();
        });

        req.on('close', () => {
            if (ws.readyState === WebSocket.OPEN) ws.close();
        });

    } catch (e) {
        console.error('[chat] error:', e.message);
        send('error', e.message);
        res.end();
    }
});

// ── Chat Sessions (log viewer) ──

app.get('/api/chat-sessions', async (_req, res) => {
    try {
        if (!existsSync(LOGS_DIR)) return res.json([]);
        const files = (await readdir(LOGS_DIR))
            .filter(f => f.endsWith('.log'))
            .sort().reverse()
            .map(f => f.replace('.log', ''));
        res.json(files);
    } catch { res.json([]); }
});

app.get('/api/chat-sessions/:sid', async (req, res) => {
    const sid = req.params.sid.replace(/[^a-zA-Z0-9_-]/g, '');
    const filePath = resolve(LOGS_DIR, `${sid}.log`);
    try {
        res.json({ content: await readFile(filePath, 'utf-8') });
    } catch { res.json({ content: 'Session not found' }); }
});

// ── Analyzers ──

app.get('/api/analyzers', async (_req, res) => {
    const manifestDir = resolve(DEPLOY_DIR, 's3_files', 'manifests');
    try {
        const names = (await readdir(manifestDir))
            .filter(f => f.endsWith('.json'))
            .map(f => f.replace('.json', ''))
            .sort();
        res.json(names);
    } catch { res.json([]); }
});

// ── Analyzer Prompts ──

app.get('/api/analyzers/:name/prompts', async (req, res) => {
    const name = req.params.name.replace(/[^a-zA-Z0-9_-]/g, '');
    const manifestPath = resolve(DEPLOY_DIR, 's3_files', 'manifests', `${name}.json`);
    const promptsBase = resolve(DEPLOY_DIR, 's3_files', 'prompts');

    try {
        const manifest = JSON.parse(await readFile(manifestPath, 'utf-8'));
        const promptFiles = manifest.analyzer?.prompt_files || [];
        const analyzerPromptDir = resolve(promptsBase, name);
        const result = {};

        for (const relPath of promptFiles) {
            // Prompt files can be relative to the analyzer dir or use ../ for shared
            const fullPath = resolve(analyzerPromptDir, relPath);
            const fileName = relPath.replace(/^\.\.\//, '');
            try {
                result[fileName] = await readFile(fullPath, 'utf-8');
            } catch {
                // File might not exist locally — skip silently
            }
        }

        res.json(result);
    } catch (e) {
        console.error(`[analyzer-prompts] error loading ${name}:`, e.message);
        res.json({});
    }
});

app.put('/api/analyzers/:name/prompts', async (req, res) => {
    const name = req.params.name.replace(/[^a-zA-Z0-9_-]/g, '');
    const promptsBase = resolve(DEPLOY_DIR, 's3_files', 'prompts');
    const analyzerPromptDir = resolve(promptsBase, name);

    try {
        const edits = req.body || {};
        for (const [fileName, content] of Object.entries(edits)) {
            // Resolve relative to analyzer dir (handles shared/ paths too)
            const fullPath = resolve(analyzerPromptDir, fileName);
            // Safety: ensure we stay within the prompts directory
            if (!fullPath.startsWith(promptsBase)) {
                console.warn(`[analyzer-prompts] path escape attempt: ${fileName}`);
                continue;
            }
            await writeFile(fullPath, content, 'utf-8');
        }
        console.log(`[analyzer-prompts] saved ${Object.keys(edits).length} files for ${name}`);
        res.json({ ok: true });
    } catch (e) {
        console.error(`[analyzer-prompts] error saving ${name}:`, e.message);
        res.status(500).json({ error: e.message });
    }
});

// ── Observability (CloudWatch aws/spans) ──

async function cwQuery(query, startTime, endTime, limit = 10000, isCancelled = () => false) {
    console.log(`[observability] query: ${query.slice(0, 120)}...`);
    console.log(`[observability] startTime=${Math.floor(startTime / 1000)} endTime=${Math.floor(endTime / 1000)}`);
    const { queryId } = await cwLogsClient.send(new StartQueryCommand({
        logGroupName: 'aws/spans',
        startTime: Math.floor(startTime / 1000),
        endTime: Math.floor(endTime / 1000),
        queryString: query,
        limit,
    }));
    console.log(`[observability] queryId=${queryId}`);
    for (let i = 0; i < 60; i++) {
        if (isCancelled()) { console.log('[observability] client disconnected, aborting poll'); return []; }
        await new Promise(r => setTimeout(r, 1000));
        const result = await cwLogsClient.send(new GetQueryResultsCommand({ queryId }));
        console.log(`[observability] poll ${i + 1}: status=${result.status} results=${(result.results || []).length}`);
        if (['Complete', 'Failed', 'Cancelled', 'Timeout'].includes(result.status)) {
            return result.results || [];
        }
    }
    console.log('[observability] query timed out after 60 polls');
    return [];
}

app.post('/api/observability', async (req, res) => {
    const { session_id } = req.body;
    if (!session_id?.trim()) return res.json({ error: 'Please enter a Session ID' });

    const sid = session_id.trim();
    const hoursBack = Math.min(Math.max(parseInt(req.body.hours_back) || 24, 1), 720);
    const endTime = Date.now();
    const startTime = endTime - hoursBack * 60 * 60 * 1000;
    console.log(`[observability] session=${sid} hoursBack=${hoursBack} startTime=${new Date(startTime).toISOString()} endTime=${new Date(endTime).toISOString()}`);

    // Track client disconnect for cancelling long polls
    let clientDisconnected = false;
    res.on('close', () => { clientDisconnected = true; });

    try {
        console.log(`[observability] fetching traces for session: ${sid}`);
        const out = [];
        out.push('='.repeat(70));
        out.push(`SESSION: ${sid}`);
        out.push('='.repeat(70), '');

        // Step 1: Find trace IDs
        let results = await cwQuery(
            `fields traceId, name, @timestamp ` +
            `| filter attributes.\`aws.bedrock.agentcore.session_id\` = '${sid}' ` +
            `   or attributes.\`session.id\` = '${sid}' ` +
            `   or attributes.\`rpc.request.metadata.x-amzn-bedrock-agentcore-runtime-session-id\` = '${sid}' ` +
            `| stats count() as cnt, min(@timestamp) as first_seen, max(@timestamp) as last_seen by traceId ` +
            `| sort first_seen asc`,
            startTime, endTime, 10000, () => clientDisconnected
        );

        if (!results.length) {
            results = await cwQuery(
                `fields traceId, name, @message | filter @message like '${sid}' | stats count() as cnt by traceId | sort cnt desc`,
                startTime, endTime, 10000, () => clientDisconnected
            );
        }

        const traceIds = [];
        for (const row of results) {
            const obj = Object.fromEntries(row.map(f => [f.field, f.value]));
            if (obj.traceId) {
                traceIds.push(obj.traceId);
                out.push(`Trace: ${obj.traceId}  spans=${obj.cnt || '?'}`);
            }
        }

        if (!traceIds.length) {
            return res.json({ output: `No traces found for session ${sid}\n\nSearched: last ${hoursBack} hours\n\nMake sure:\n1. The 'aws/spans' log group exists\n2. The session is within the selected time range` });
        }

        out.push(`\nFound ${traceIds.length} trace(s)`, '');

        // Step 2: Fetch spans
        const allSpans = [];
        for (const tid of traceIds) {
            const spanResults = await cwQuery(
                `fields @message | filter traceId = '${tid}' | sort @timestamp asc | limit 10000`,
                startTime, endTime, 10000, () => clientDisconnected
            );
            for (const row of spanResults) {
                for (const f of row) {
                    if (f.field === '@message') {
                        try { const doc = JSON.parse(f.value); doc._traceId = tid; allSpans.push(doc); } catch { }
                    }
                }
            }
        }

        out.push(`Total Spans: ${allSpans.length}`);

        // Step 3: Extract events and stats
        const allEvents = [];
        const spanWithEvents = {};
        let totalInput = 0, totalOutput = 0;
        const models = new Set();
        const toolsList = [];

        for (const doc of allSpans) {
            const name = doc.name || '?';
            const attrs = doc.attributes || {};
            const events = doc.events || [];

            if (attrs['gen_ai.usage.input_tokens']) totalInput = Math.max(totalInput, +attrs['gen_ai.usage.input_tokens']);
            if (attrs['gen_ai.usage.output_tokens']) totalOutput = Math.max(totalOutput, +attrs['gen_ai.usage.output_tokens']);
            if (attrs['gen_ai.request.model']) models.add(attrs['gen_ai.request.model']);

            if (events.length) spanWithEvents[name] = (spanWithEvents[name] || 0) + events.length;

            for (const evt of events) {
                const evtAttrs = evt.attributes || {};
                const content = evtAttrs.content || evtAttrs.message || '';
                if (typeof content === 'string' && content.includes('toolUse')) {
                    const matches = content.match(/"name":\s*"([^"]+)"/g) || [];
                    toolsList.push(...matches.map(m => m.match(/"name":\s*"([^"]+)"/)[1]));
                }

                let ts = '';
                try { ts = new Date(+evt.timeUnixNano / 1e6).toISOString().slice(11, 23); } catch { }

                allEvents.push({
                    traceId: doc._traceId, spanName: name, spanId: doc.spanId || '',
                    eventName: evt.name || 'unnamed', eventTime: ts,
                    eventTimeNs: evt.timeUnixNano || '0', eventAttrs: evtAttrs,
                });
            }
        }

        allEvents.sort((a, b) => (BigInt(a.eventTimeNs) < BigInt(b.eventTimeNs) ? -1 : 1));

        out.push(`Total Events: ${allEvents.length}`, '');
        out.push('── Token Usage ──');
        out.push(`  Input:  ${totalInput.toLocaleString()}`);
        out.push(`  Output: ${totalOutput.toLocaleString()}`);
        out.push(`  Total:  ${(totalInput + totalOutput).toLocaleString()}`, '');

        if (models.size) {
            out.push('── Models Used ──');
            for (const m of [...models].sort()) out.push(`  • ${m}`);
            out.push('');
        }

        if (toolsList.length) {
            out.push('── Tools Called ──');
            const counts = {};
            toolsList.forEach(t => counts[t] = (counts[t] || 0) + 1);
            Object.entries(counts).sort((a, b) => b[1] - a[1]).forEach(([t, c]) => out.push(`  ${t}: ${c}x`));
            out.push('');
        }

        out.push('── Event Types ──');
        const evtCounts = {};
        allEvents.forEach(e => evtCounts[e.eventName] = (evtCounts[e.eventName] || 0) + 1);
        Object.entries(evtCounts).sort((a, b) => b[1] - a[1]).forEach(([n, c]) => out.push(`  ${String(c).padStart(4)}  ${n}`));
        out.push('');

        out.push('── Span Types (with events) ──');
        Object.entries(spanWithEvents).sort((a, b) => b[1] - a[1]).slice(0, 15).forEach(([n, c]) => out.push(`  ${String(c).padStart(4)}  ${n}`));
        out.push('');

        // Event timeline
        out.push('='.repeat(70));
        out.push(`EVENT TIMELINE (${allEvents.length} events)`);
        out.push('='.repeat(70));

        allEvents.slice(0, 150).forEach((evt, i) => {
            const content = evt.eventAttrs.content || evt.eventAttrs.message || evt.eventAttrs.body || '';
            let summary = '';
            if (typeof content === 'string') {
                if (content.includes('toolUse')) {
                    const m = content.match(/"name":\s*"([^"]+)"/g) || [];
                    summary = `tools=[${m.map(x => x.match(/"name":\s*"([^"]+)"/)[1]).join(', ')}]`;
                } else if (content.includes('toolResult')) {
                    summary = '[tool result]';
                } else if (content.length > 120) {
                    summary = content.slice(0, 120) + '...';
                } else if (content) {
                    summary = content;
                }
            }
            if (!summary && Object.keys(evt.eventAttrs).length) {
                summary = JSON.stringify(evt.eventAttrs).slice(0, 120);
            }

            out.push(`\n[${String(i + 1).padStart(3)}] ${evt.eventTime}  ${evt.eventName}`);
            out.push(`      span: ${evt.spanName}`);
            if (summary) out.push(`      ${summary}`);
        });

        if (allEvents.length > 150) out.push(`\n... and ${allEvents.length - 150} more events`);

        res.json({ output: out.join('\n') });
    } catch (e) {
        console.error('[observability] error:', e.message);
        res.json({ error: e.message });
    }
});

// ── Pricing Config ──

app.get('/api/pricing-config', async (_req, res) => {
    const configPath = resolve(LOCAL_TESTING_DIR, 'config', 'pricing_config.json');
    try {
        const data = JSON.parse(await readFile(configPath, 'utf-8'));
        res.json(data);
    } catch (e) {
        res.status(500).json({ error: 'Failed to load pricing config: ' + e.message });
    }
});

// ── Stub endpoints for wizard/evaluator ──

app.post('/api/wizard/generate', (_req, res) => res.json({ prompts: {} }));
app.post('/api/wizard/preview', (_req, res) => res.json({}));
app.post('/api/wizard/deploy', (_req, res) => res.json({ output: 'Not yet wired' }));
// ── Evaluator: S3-backed session evaluation ──

async function s3GetJson(bucket, key) {
    const resp = await s3Client.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
    return JSON.parse(await resp.Body.transformToString());
}

app.get('/api/eval/sessions', async (_req, res) => {
    if (!OUTPUT_BUCKET) return res.json([]);
    try {
        const resp = await s3Client.send(new ListObjectsV2Command({ Bucket: OUTPUT_BUCKET, Delimiter: '/' }));
        const sessions = (resp.CommonPrefixes || [])
            .map(p => p.Prefix.replace(/\/$/, ''))
            .filter(s => s !== 'evaluations' && s !== 'temp')
            .sort().reverse();
        res.json(sessions);
    } catch (e) { console.error('[eval] list sessions:', e.message); res.json([]); }
});

app.get('/api/eval/sessions/:sid', async (req, res) => {
    if (!OUTPUT_BUCKET) return res.json({ results: [], metadata: null, ratings: {} });
    const sid = req.params.sid.replace(/[^a-zA-Z0-9_-]/g, '');
    try {
        // List all XML result files in session (recursive)
        const results = [];
        let token;
        do {
            const resp = await s3Client.send(new ListObjectsV2Command({
                Bucket: OUTPUT_BUCKET, Prefix: `${sid}/`, ContinuationToken: token,
            }));
            for (const obj of resp.Contents || []) {
                if (obj.Key.endsWith('.xml') && !obj.Key.includes('/evaluations/')) {
                    const parts = obj.Key.split('/');
                    const filename = parts[parts.length - 1];
                    const analyzer = parts.length >= 3 ? parts[1] : filename.split('_')[0] + '_analyzer';
                    results.push({ key: obj.Key, filename, analyzer, size: obj.Size });
                }
            }
            token = resp.NextContinuationToken;
        } while (token);

        // Session metadata
        let metadata = null;
        try { metadata = await s3GetJson(OUTPUT_BUCKET, `${sid}/session_metadata.json`); } catch { }

        // Existing evaluations (session-level file)
        let ratings = {};
        try {
            const evalData = await s3GetJson(OUTPUT_BUCKET, `${sid}/evaluations/session_evaluation.json`);
            for (const ev of evalData.evaluations || []) {
                if (ev.result_file) ratings[ev.result_file] = ev.responses || {};
            }
        } catch { }

        res.json({ results, metadata, ratings });
    } catch (e) { console.error('[eval] load session:', e.message); res.json({ results: [], metadata: null, ratings: {} }); }
});

app.get('/api/eval/result', async (req, res) => {
    if (!OUTPUT_BUCKET) return res.json({ content: '' });
    const key = req.query.key;
    if (!key) return res.json({ content: '' });
    try {
        const resp = await s3Client.send(new GetObjectCommand({ Bucket: OUTPUT_BUCKET, Key: key }));
        res.json({ content: await resp.Body.transformToString() });
    } catch (e) { res.json({ content: `Error: ${e.message}` }); }
});

app.get('/api/eval/manifest-eval/:analyzer', async (req, res) => {
    if (!CONFIG_BUCKET) return res.json({ evaluation: null });
    const name = req.params.analyzer.replace(/[^a-zA-Z0-9_-]/g, '');
    for (const prefix of ['manifests', 'custom-analyzers/manifests']) {
        try {
            const manifest = await s3GetJson(CONFIG_BUCKET, `${prefix}/${name}.json`);
            return res.json({ evaluation: manifest.evaluation || null });
        } catch { }
    }
    res.json({ evaluation: null });
});

app.put('/api/eval/sessions/:sid/ratings', async (req, res) => {
    if (!OUTPUT_BUCKET) return res.json({ ok: false });
    const sid = req.params.sid.replace(/[^a-zA-Z0-9_-]/g, '');
    const { result_file, analyzer, responses } = req.body;
    try {
        const evalKey = `${sid}/evaluations/session_evaluation.json`;
        let existing = { evaluations: [] };
        try { existing = await s3GetJson(OUTPUT_BUCKET, evalKey); } catch { }
        if (!existing.evaluations) existing.evaluations = [];

        const now = new Date();
        const entry = {
            result_file, analyzer, responses,
            evaluated_at: now.toISOString(),
            evaluated_at_readable: now.toISOString().slice(0, 16),
        };
        const idx = existing.evaluations.findIndex(e => e.result_file === result_file);
        if (idx >= 0) existing.evaluations[idx] = entry;
        else existing.evaluations.push(entry);
        existing.last_updated = now.toISOString();

        await s3Client.send(new PutObjectCommand({
            Bucket: OUTPUT_BUCKET, Key: evalKey,
            Body: JSON.stringify(existing, null, 2), ContentType: 'application/json',
        }));
        res.json({ ok: true });
    } catch (e) { console.error('[eval] save:', e.message); res.json({ ok: false, error: e.message }); }
});

// ── Start ──

const PORT = process.env.PORT || 3457;
app.listen(PORT, () => {
    console.log(`\n🦡 BADGERS Local Testing API on http://localhost:${PORT}\n`);
});
