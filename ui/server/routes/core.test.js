import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test, { after } from 'node:test';
import { DynamoDBDocumentClient } from '@aws-sdk/lib-dynamodb';
import { mountCoreRoutes } from './core.js';

const tempRoot = mkdtempSync(join(tmpdir(), 'badgers-core-routes-'));
process.env.CHAT_LOGS_DIR = join(tempRoot, 'chat_sessions');
process.env.JOBS_TABLE_NAME = 'test-jobs';

const routes = new Map();
const app = {
    get(path, handler) { routes.set(`GET ${path}`, handler); },
    post(path, handler) { routes.set(`POST ${path}`, handler); },
    put(path, handler) { routes.set(`PUT ${path}`, handler); },
    delete(path, handler) { routes.set(`DELETE ${path}`, handler); },
};

mountCoreRoutes(app, tempRoot);

function ownerDir(subject) {
    if (subject === 'local') return 'local';
    return createHash('sha256').update(subject).digest('hex').slice(0, 24);
}

function response() {
    return {
        statusCode: 200,
        body: undefined,
        status(code) { this.statusCode = code; return this; },
        json(body) { this.body = body; return this; },
    };
}

async function invoke(path, req) {
    const handler = routes.get(path);
    assert.ok(handler, `route ${path} is registered`);
    const res = response();
    await handler(req, res);
    return res;
}

after(() => rmSync(tempRoot, { recursive: true, force: true }));

test('chat transcript routes are scoped to the authenticated caller', async () => {
    const logRoot = process.env.CHAT_LOGS_DIR;
    const aliceDir = join(logRoot, ownerDir('alice-sub'));
    const bobDir = join(logRoot, ownerDir('bob-sub'));
    mkdirSync(aliceDir, { recursive: true });
    mkdirSync(bobDir, { recursive: true });
    writeFileSync(join(aliceDir, 'alice-session.log'), 'alice private transcript\n');
    writeFileSync(join(bobDir, 'bob-session.log'), 'bob private transcript\n');

    const list = await invoke('GET /api/chat-sessions', { user: { sub: 'alice-sub' } });
    assert.equal(list.statusCode, 200);
    assert.deepEqual(list.body, ['alice-session']);

    const foreignRead = await invoke('GET /api/chat-sessions/:sid', {
        user: { sub: 'alice-sub' },
        params: { sid: 'bob-session' },
    });
    assert.equal(foreignRead.statusCode, 404);
    assert.deepEqual(foreignRead.body, { error: 'Session not found' });

    const foreignTrace = await invoke('POST /api/observability', {
        user: { sub: 'alice-sub' },
        body: { session_id: 'bob-session' },
    });
    assert.equal(foreignTrace.statusCode, 404);
    assert.deepEqual(foreignTrace.body, { error: 'Session not found' });
});

test('job routes require the owner on the orchestrator record', async () => {
    const originalSend = DynamoDBDocumentClient.prototype.send;
    DynamoDBDocumentClient.prototype.send = async function send(command) {
        const input = command.input;
        if (input.IndexName === 'owner-index') {
            // A caller with a different Cognito subject has no candidate jobs,
            // even when they know another caller's doc_id.
            return { Items: input.ExpressionAttributeValues[':owner'] === 'alice-sub' ? [{ job_id: 'alice-job' }] : [] };
        }
        if (input.ExpressionAttributeValues?.[':jid'] === 'alice-job') {
            return {
                Items: [
                    { job_id: 'alice-job', subtask_id: 'orchestrator', owner_sub: 'alice-sub', doc_id: 'shared-doc', status: 'PENDING' },
                    { job_id: 'alice-job', subtask_id: 'page#1', status: 'COMPLETE', specialist: 'page' },
                ],
            };
        }
        return { Items: [] };
    };

    try {
        const foreignJob = await invoke('GET /api/jobs/:jobId', {
            user: { sub: 'bob-sub' },
            params: { jobId: 'alice-job' },
        });
        assert.equal(foreignJob.statusCode, 404);
        assert.deepEqual(foreignJob.body, { error: 'Job not found' });

        const foreignDocJobs = await invoke('GET /api/jobs', {
            user: { sub: 'bob-sub' },
            query: { doc_id: 'shared-doc' },
        });
        assert.equal(foreignDocJobs.statusCode, 200);
        assert.deepEqual(foreignDocJobs.body, { doc_id: 'shared-doc', jobs: [] });
    } finally {
        DynamoDBDocumentClient.prototype.send = originalSend;
    }
});
