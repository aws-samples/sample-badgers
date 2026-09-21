// GET /api/models — the models this deployment can actually invoke.
//
// Joins two sources, because neither alone is sufficient:
//
//   SSM  /badgers-{id}/model-profiles   what is provisioned right now, written by
//                                       InferenceProfilesStack in the same loop that
//                                       creates the profiles
//   S3   config/model_registry.json     display names, prices, capability flags
//
// A model is returned only if it appears in BOTH and its registry status is "active".
// SSM is the authority on existence, so a model added to the registry but not yet deployed
// degrades to a missing dropdown entry rather than a specialist that saves cleanly and
// fails on first invocation.
//
// This replaces two hand-maintained lists — MODEL_IDS in wizard.js and MODELS in
// SpecialistWizard.jsx — which disagreed with each other and with AWS.

import { GetParameterCommand, SSMClient } from '@aws-sdk/client-ssm';
import { GetObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { fromNodeProviderChain } from '@aws-sdk/credential-providers';

// Short TTL rather than load-once. The map changes on deploy, and a UI task outliving a
// deploy would otherwise keep offering the previous model set until restarted.
const CACHE_TTL_MS = 60_000;

let cache = { at: 0, models: null };

// Presentation order for the wizard: workhorse, then heavy reasoning, then ascending price.
const RANK_FIRST = [
    'us.anthropic.claude-sonnet-4-6',
    'us.anthropic.claude-opus-4-6-v1',
    'us.anthropic.claude-opus-5',
];

/**
 * The composite `{id}-{suffix}` that names every per-deployment resource, including the
 * SSM prefix `/badgers-{id}-{suffix}/`.
 *
 * Two environments spell it differently. The ECS task definition sets DEPLOYMENT_ID to the
 * composite already ("dev-a1b") and STACK_SUFFIX to the bare suffix. ui/.env, written by
 * generate_ui_env.sh for local development, sets DEPLOYMENT_ID to the bare id ("dev") and
 * STACK_SUFFIX separately. Reading DEPLOYMENT_ID alone therefore works in the container
 * and misses every parameter locally. Append the suffix only when it is not already there.
 */
export function resourceId(ENV = process.env) {
    const id = ENV.DEPLOYMENT_ID || '';
    const suffix = ENV.STACK_SUFFIX || '';
    if (!id) return '';
    if (!suffix || id.endsWith(`-${suffix}`)) return id;
    return `${id}-${suffix}`;
}

async function readProfileMap(ssmClient, deploymentId) {
    if (!deploymentId) throw new Error('DEPLOYMENT_ID is not set');
    const name = `/badgers-${deploymentId}/model-profiles`;
    const resp = await ssmClient.send(new GetParameterCommand({ Name: name }));
    const parsed = JSON.parse(resp.Parameter?.Value || '{}');
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
        throw new Error(`${name} is not a JSON object`);
    }
    return parsed;
}

async function readRegistry(s3Client, bucket) {
    if (!bucket) throw new Error('S3_CONFIG_BUCKET is not set');
    const resp = await s3Client.send(new GetObjectCommand({
        Bucket: bucket, Key: 'config/model_registry.json',
    }));
    const parsed = JSON.parse(await resp.Body.transformToString());
    if (!parsed?.models) throw new Error('model_registry.json has no "models" key');
    return parsed.models;
}

/**
 * Resolve the deployed model set, cached for CACHE_TTL_MS.
 *
 * Exported so the wizard validates a submitted model ID against the same join the dropdown
 * was populated from, rather than against a second list that can drift.
 */
export async function listModels(ENV = process.env) {
    if (cache.models && Date.now() - cache.at < CACHE_TTL_MS) return cache.models;

    const region = ENV.AWS_REGION || 'us-west-2';
    const credentials = fromNodeProviderChain({ profile: ENV.AWS_PROFILE || undefined });
    const ssmClient = new SSMClient({ region, credentials });
    const s3Client = new S3Client({ region, credentials });

    // Read env per call, not at import: routes mount before index.js awaits loadSSMConfig(),
    // so a value arriving from SSM would otherwise be missed.
    const [profiles, registry] = await Promise.all([
        readProfileMap(ssmClient, resourceId(ENV)),
        readRegistry(s3Client, ENV.S3_CONFIG_BUCKET || ''),
    ]);

    // A model is offered when it is active AND deployed. "Deployed" means it has an
    // application inference profile in the SSM map — except mantle models, which have no
    // profile by design (they attribute cost through the Bedrock default project), so their
    // presence in the active registry is sufficient.
    const models = Object.entries(registry)
        .filter(([modelId, spec]) =>
            spec.status === 'active' &&
            (modelId in profiles || spec.transport === 'mantle'))
        .map(([modelId, spec]) => ({
            model_id: modelId,
            display_name: spec.display_name,
            provider: spec.provider,
            price_in: spec.price_in,
            price_out: spec.price_out,
            thinking: spec.thinking ?? null,
            prompt_caching: Boolean(spec.prompt_caching),
        }));

    const rank = (m) => {
        const i = RANK_FIRST.indexOf(m.model_id);
        return i === -1 ? RANK_FIRST.length : i;
    };
    models.sort((a, b) => rank(a) - rank(b) || a.price_in - b.price_in);

    cache = { at: Date.now(), models };
    return models;
}

export function mountModelsRoutes(app, ENV = process.env) {
    app.get('/api/models', async (_req, res) => {
        try {
            const models = await listModels(ENV);
            res.json({ models });
        } catch (e) {
            // No fallback to a hardcoded list. A stale list is what this endpoint exists to
            // remove, and serving one would let the wizard offer a model with no profile —
            // which fails at invocation, not at save.
            console.error('GET /api/models failed:', e.message);
            res.status(503).json({
                error: `Could not resolve the deployed model set: ${e.message}`,
                models: [],
            });
        }
    });
}

export function _clearModelsCache() {
    cache = { at: 0, models: null };
}
