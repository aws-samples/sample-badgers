// Specialist creation wizard.
//
// Self-contained: everything this module writes lands under
// deployment/custom_specialists/, which the CustomSpecialistsStack already reads.
// Nothing here touches the built-in specialists in deployment/s3_files/.
//
// Four endpoints, matching the wizard's four steps:
//   POST /api/wizard/generate  -> six Bedrock calls, one per prompt section
//   POST /api/wizard/preview   -> assemble manifest + schema, return for display
//   POST /api/wizard/save      -> write artifacts to disk (fast, plain JSON)
//   POST /api/wizard/deploy    -> stream `deploy_custom_specialists.sh` over SSE
//
// Bedrock is called by signing a plain HTTPS request rather than via
// @aws-sdk/client-bedrock-runtime, which is not a dependency of this package.
// The signing primitives below are already used elsewhere in the server, so
// this adds no new dependency and no change to the container build.

import { existsSync, mkdirSync } from 'fs';
import { readFile, writeFile } from 'fs/promises';
import { resolve, extname } from 'path';
import { spawn } from 'child_process';
import { SignatureV4 } from '@smithy/signature-v4';
import { HttpRequest } from '@smithy/protocol-http';
import { Sha256 } from '@aws-crypto/sha256-js';
import { fromNodeProviderChain } from '@aws-sdk/credential-providers';
import { listModels } from './models.js';

// The six prompt sections a specialist is built from, in load order.
const PROMPT_TYPES = ['gestalt', 'job_role', 'context', 'rules', 'tasks', 'format'];

// Few-shot source for each section, taken from specialists that already ship.
// Paths are relative to deployment/s3_files/prompts/.
const EXAMPLE_MAP = {
    gestalt: ['elements_specialist', 'elements_gestalt.xml'],
    job_role: ['elements_specialist', 'elements_job_role.xml'],
    context: ['correlation_specialist', 'correlation_context.xml'],
    rules: ['correlation_specialist', 'correlation_rules.xml'],
    tasks: ['elements_specialist', 'elements_tasks_extraction.xml'],
    format: ['elements_specialist', 'elements_format.xml'],
};

// Human-readable label and one-line purpose for each section, shown in the UI
// while that section is being generated.
const PROMPT_META = {
    gestalt: { label: 'Gestalt Perception', purpose: 'Holistic visual scan of the page before any extraction.' },
    job_role: { label: 'Job Role', purpose: 'The expert persona the model adopts.' },
    context: { label: 'Context', purpose: 'What counts as a target element, and what does not.' },
    rules: { label: 'Rules', purpose: 'Anti-hallucination and two-pass verification constraints.' },
    tasks: { label: 'Tasks', purpose: 'The step-by-step extraction procedure.' },
    format: { label: 'Response Format', purpose: 'The exact output structure the model must return.' },
};

// Per-section generation instructions. Carried over from the previous
// implementation of this wizard, which had these tuned already.
const PROMPT_TYPE_INSTRUCTIONS = {
    gestalt:
        'Generate a gestalt_perception XML prompt that applies Gestalt perception principles ' +
        'to the target document/content type. Include:\n' +
        '- A <prescan> section with 5-6 steps for holistic visual scanning before extraction\n' +
        '- A <principles_emphasis> section with 2-3 primary/secondary Gestalt principles ' +
        '(proximity, similarity, continuity, closure, figure_ground, common_fate) with specific ' +
        "guidance on how each applies to this specialist's domain\n" +
        '- An <element_detection_cues> section with 5-8 visual cues specific to the content type\n' +
        '- A <validation> section with 4-5 checks to verify perception accuracy after extraction\n' +
        "The prompt must guide the Vision LLM to 'see' the document holistically before extracting.",
    job_role:
        "Generate a job_role XML prompt that establishes the LLM's expert persona. Include:\n" +
        '- A <role> tag with a specialist title and one-sentence expertise statement\n' +
        '- A <job_description> with title, 2-3 sentence summary, 5-7 responsibilities, and 5-7 skills\n' +
        'All responsibilities and skills must be specific to the analysis domain, not generic.',
    context:
        'Generate a context XML prompt with 10-12 <item> elements. Include:\n' +
        '- What the specialist searches for and extracts\n' +
        '- Clear definition of what IS the target element type\n' +
        '- Clear definition of what is NOT the target element type (common misidentifications)\n' +
        '- Variations and forms the target element may take\n' +
        '- Two-step verification reminder\n' +
        '- Guidance on prioritizing accuracy over comprehensiveness\n' +
        '- What to do when nothing is found (not_found response)',
    rules:
        'Generate a rules XML prompt with 8-12 rules. Order them as:\n' +
        "1. Anti-hallucination rules first (do not invent, do not create elements that don't exist)\n" +
        '2. Two-step verification rule (PASS 1 identify, PASS 2 verify)\n' +
        '3. Format compliance rules (use only the response_format provided)\n' +
        '4. Honest negatives rule (finding nothing is always valid)\n' +
        '5. Domain-specific rules last\n' +
        'Rules may include a priority attribute (critical, high, medium, low).',
    tasks:
        'Generate a tasks XML prompt with 12-16 <task> elements. Tasks may contain <sub_task> ' +
        'children for multi-step instructions. Structure as:\n' +
        '- Initial review/deep-breath task\n' +
        '- PASS 1 tasks: scan and identify potential elements by visual/textual indicators\n' +
        '- PASS 2 tasks: verify each candidate against strict criteria, reject false positives\n' +
        '- Extraction tasks: for verified elements, extract specific data points\n' +
        '- Organization task: structure findings per the response_format\n' +
        '- Final review task: verify accuracy, completeness, correct ordering\n' +
        'Tasks should be detailed and actionable, not one-liners.',
    format:
        'Generate a response_format XML prompt that defines the exact output structure. Include:\n' +
        '- A <response> wrapper with extraction_type attribute\n' +
        '- A <metadata> section with page_number, examples_count, element_count\n' +
        '- An <elements> section with one <element> template showing all fields to extract, ' +
        'using {SCREAMING_SNAKE_CASE} placeholder tokens for values the LLM fills in\n' +
        '- A <not_found> section with a message template for when no elements are detected\n' +
        '- A comment noting to omit <elements> when element_count is 0',
};

// Model used to write the prompts themselves, when nothing else says. Independent
// of the model the generated specialist will run on. The resolved value is read
// inside mountWizardRoutes: in ECS the task definition sets WIZARD_GENERATOR_MODEL_ID
// from the same constant the task role's Bedrock grant was built from
// (deployment/stacks/ecs_stack.py), so the call and the grant cannot name different
// models. This literal is what local development uses.
const DEFAULT_GENERATOR_MODEL_ID = 'us.anthropic.claude-sonnet-4-6';

// The client now submits model IDs, not display labels, because the dropdown is populated
// from GET /api/models. The label -> ID map that used to live here is gone: it was a second
// hand-maintained list that disagreed with AWS in three ways (Sonnet 4.6 spelled with a
// -v1:0 suffix the CDK never creates; Nova Pro/Lite/Micro mappable despite never having had
// a profile; several entries naming retired models).
//
// Validation is against the same join the dropdown was built from, so the two cannot drift.
async function assertModelsAvailable(ids) {
    const wanted = ids.filter(Boolean);
    if (wanted.length === 0) return;

    const available = await listModels();
    const known = new Set(available.map((m) => m.model_id));
    const unknown = wanted.filter((id) => !known.has(id));

    if (unknown.length) {
        throw new Error(
            `Not available in this deployment: ${unknown.join(', ')}. ` +
            `Available: ${[...known].join(', ')}. A model must be 'active' in the registry ` +
            `and have a provisioned inference profile.`
        );
    }
}

// "Paleographic Specialist" -> "paleographic_specialist"
// Always ends in exactly one _specialist suffix, since the stack, the manifest
// filename, and the Lambda name all key off this single value.
function sanitizeName(displayName) {
    let base = String(displayName || '')
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '_')
        // The collapse above guarantees no '__' run, so leading/trailing
        // underscores are at most one char. Strip a single one at each end
        // (no '+' quantifier) to avoid a polynomial-backtracking regex on
        // attacker-controlled input (ReDoS).
        .replace(/^_/, '')
        .replace(/_$/, '');
    if (!base) return '';
    while (base.endsWith('_specialist')) base = base.slice(0, -'_specialist'.length).replace(/_$/, '');
    return base ? `${base}_specialist` : '';
}

function shortNameOf(specialistName) {
    return specialistName.endsWith('_specialist')
        ? specialistName.slice(0, -'_specialist'.length)
        : specialistName;
}

// Models sometimes wrap output in a code fence or lead with a sentence of
// preamble despite being told not to. Strip both and start at the first tag.
function extractXml(text) {
    let out = String(text || '').trim();
    if (!out) return out;
    if (out.includes('```xml')) {
        out = out.split('```xml')[1];
        if (out.includes('```')) out = out.split('```')[0];
    } else if (out.includes('```')) {
        out = out.split('```')[1] ?? '';
        if (out.includes('```')) out = out.split('```')[0];
    }
    out = out.trim();
    if (!out.startsWith('<')) {
        const i = out.indexOf('<');
        if (i >= 0) out = out.slice(i);
    }
    return out.trim();
}

export function mountWizardRoutes(app, PROJECT_ROOT) {
    const DEPLOY_DIR = resolve(PROJECT_ROOT, 'deployment');
    const CUSTOM_DIR = resolve(DEPLOY_DIR, 'custom_specialists');
    const BUILTIN_PROMPTS_DIR = resolve(DEPLOY_DIR, 's3_files', 'prompts');

    const REGION = process.env.AWS_REGION || 'us-west-2';
    const AWS_PROFILE = process.env.AWS_PROFILE || undefined;
    const credentials = fromNodeProviderChain({ profile: AWS_PROFILE });
    const GENERATOR_MODEL_ID = process.env.WIZARD_GENERATOR_MODEL_ID || DEFAULT_GENERATOR_MODEL_ID;

    // Guard every write: resolve the final path and confirm it is still inside
    // custom_specialists/ before touching the filesystem.
    function safePath(...parts) {
        const target = resolve(CUSTOM_DIR, ...parts);
        if (target !== CUSTOM_DIR && !target.startsWith(CUSTOM_DIR + '/')) {
            throw new Error('Path traversal blocked');
        }
        return target;
    }

    async function loadFewShot(promptType) {
        const mapping = EXAMPLE_MAP[promptType];
        if (!mapping) return null;
        const path = resolve(BUILTIN_PROMPTS_DIR, mapping[0], mapping[1]);
        try {
            let content = await readFile(path, 'utf-8');
            if (content.startsWith('<?xml')) content = content.split('?>').slice(1).join('?>');
            return content.trim();
        } catch {
            return null;
        }
    }

    function buildSystemPrompt(promptType, fewShot) {
        const parts = [
            'You are an expert at creating XML prompts for document analysis systems.',
            `You are generating ONLY the ${promptType} prompt for a new specialist.`,
            '',
            '## Instructions',
            PROMPT_TYPE_INSTRUCTIONS[promptType] || '',
            '',
            '## Critical Rules',
            '- Gestalt-First Perception: guide the model to perceive holistically before extracting.',
            '- Two-Pass Verification: PASS 1 identifies potential elements, PASS 2 verifies against strict criteria.',
            '- Honest Negatives: finding nothing is ALWAYS valid. Never force identification.',
            "- Anti-Hallucination: never create, invent, or infer elements that don't exist in the source.",
            '- Use {SCREAMING_SNAKE_CASE} placeholder tokens for values the LLM fills in during analysis.',
            '',
        ];
        if (fewShot) {
            parts.push(
                '## Reference Example',
                `Here is a high-quality example of a ${promptType} prompt from an existing specialist. ` +
                'Use it as a reference for structure, depth, and quality — but adapt the content ' +
                "to the new specialist's domain:",
                '',
                fewShot,
                ''
            );
        }
        parts.push(
            '## Output',
            'Return ONLY the raw XML content. No markdown fencing, no preamble, no commentary.',
            'Start your response with the opening XML tag.'
        );
        return parts.join('\n');
    }

    // Bedrock Converse over a hand-signed HTTPS request.
    async function converse({ modelId, system, userText, maxTokens = 8000, temperature = 0.3 }) {
        const hostname = `bedrock-runtime.${REGION}.amazonaws.com`;
        const path = `/model/${encodeURIComponent(modelId)}/converse`;
        const body = JSON.stringify({
            messages: [{ role: 'user', content: [{ text: userText }] }],
            system: [{ text: system }],
            inferenceConfig: { maxTokens, temperature },
        });

        const signer = new SignatureV4({ credentials, region: REGION, service: 'bedrock', sha256: Sha256 });
        const signed = await signer.sign(
            new HttpRequest({
                method: 'POST',
                protocol: 'https:',
                hostname,
                path,
                headers: { host: hostname, 'content-type': 'application/json' },
                body,
            })
        );

        const res = await fetch(`https://${hostname}${path}`, {
            method: 'POST',
            headers: signed.headers,
            body,
        });
        const text = await res.text();
        if (!res.ok) throw new Error(`Bedrock ${res.status}: ${text.slice(0, 400)}`);
        const data = JSON.parse(text);
        return data?.output?.message?.content?.[0]?.text ?? '';
    }

    function buildManifest({ specialistName, description, details, primaryModel, fallback1, fallback2, enhancement, exampleCount = 0 }) {
        const short = shortNameOf(specialistName);
        const manifest = {
            tool: {
                name: `analyze_${short}_tool`,
                description,
                inputSchema: {
                    type: 'object',
                    properties: {
                        image_path: {
                            type: 'string',
                            description: 'Absolute file path to the image to analyze. Supports common image formats (PNG, JPG, etc.)',
                        },
                        aws_profile: {
                            type: 'string',
                            description: 'Optional AWS profile name for Bedrock authentication. If not provided, uses default AWS credentials',
                        },
                        session_id: { type: 'string', description: 'Runtime session ID' },
                        audit_mode: {
                            type: 'boolean',
                            description: 'When true, includes confidence scoring and human review flags in the analysis output',
                        },
                    },
                    required: ['image_path', 'session_id'],
                },
            },
            specialist: {
                name: specialistName,
                description,
                prompt_specialist_prompt_base_path: 'prompts',
                prompt_files: PROMPT_TYPES.map(t => `${short}_${t}.xml`),
                examples_path: 'prompts',
                max_examples: exampleCount,
                analysis_text: (details || description || short).toString().trim().slice(0, 120),
                expected_output_tokens: 2500,
                model_selections: {
                    primary: primaryModel,
                    fallback_list: [fallback1, fallback2].filter(Boolean),
                },
                output_extension: 'xml',
            },
            metadata: {
                version: '1.0.0',
                dependencies: ['boto3'],
                wizard_managed: true,
                last_modified: new Date().toISOString(),
                specialist_type: 'standard',
            },
        };
        if (enhancement) manifest.specialist.enhancement_eligible = true;
        return manifest;
    }

    // The gateway reads this as the tool schema. It is a single-element list,
    // and `required` is session_id only, matching the built-in schemas.
    function buildSchema({ specialistName, description, displayName }) {
        const short = shortNameOf(specialistName);
        return [
            {
                name: `analyze_${short}_tool`,
                description,
                inputSchema: {
                    type: 'object',
                    properties: {
                        image_path: {
                            type: 'string',
                            description: 'S3 URL (s3://bucket/key) or absolute file path to the image. Use this OR image_data, not both.',
                        },
                        image_data: {
                            type: 'string',
                            description: 'Base64-encoded image data. Use this OR image_path, not both. Preferred when image is already in memory.',
                        },
                        aws_profile: {
                            type: 'string',
                            description: 'Optional AWS profile name for Bedrock authentication. If not provided, uses default AWS credentials',
                        },
                        session_id: { type: 'string', description: 'Runtime session ID' },
                    },
                    required: ['session_id'],
                },
                outputSchema: {
                    type: 'object',
                    properties: {
                        result: { type: 'string', description: `Analysis result from ${displayName || specialistName}` },
                        success: { type: 'boolean', description: 'Whether the analysis completed successfully' },
                    },
                },
            },
        ];
    }

    // ── Step 1: generate the six prompts ──
    //
    // Streams over SSE so the UI can name each section as it is worked on. Six
    // sequential Bedrock calls take a few minutes in total, which is too long to
    // hold a plain POST open behind a load balancer.

    app.post('/api/wizard/generate', async (req, res) => {
        const { displayName, description, details } = req.body || {};

        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.flushHeaders?.();
        const send = (obj) => { try { res.write(`data: ${JSON.stringify(obj)}\n\n`); } catch { } };
        const fail = (message) => { send({ type: 'error', message }); res.end(); };

        if (!displayName || !description) {
            return fail('Specialist name and short description are required');
        }
        const specialistName = sanitizeName(displayName);
        if (!specialistName) {
            return fail('Specialist name must contain at least one letter or number');
        }

        // Client navigated away or hit stop; abandon the remaining sections.
        let aborted = false;
        res.on('close', () => { aborted = true; });

        const total = PROMPT_TYPES.length;
        const prompts = {};
        const failures = [];

        // Ship the whole section list up front so the UI can render one
        // checkpoint per prompt before any of them have run, without keeping its
        // own copy of the list in sync with this one.
        send({
            type: 'start',
            specialistName,
            total,
            sections: PROMPT_TYPES.map(t => ({ promptType: t, ...(PROMPT_META[t] || { label: t, purpose: '' }) })),
        });

        for (let i = 0; i < total; i++) {
            if (aborted) return;
            const promptType = PROMPT_TYPES[i];
            const meta = PROMPT_META[promptType] || { label: promptType, purpose: '' };
            send({
                type: 'progress',
                index: i + 1,
                total,
                promptType,
                label: meta.label,
                purpose: meta.purpose,
            });
            try {
                const fewShot = await loadFewShot(promptType);
                const userText =
                    `Create the ${promptType} XML prompt for a new specialist with these details:\n\n` +
                    `Specialist name: ${specialistName}\n` +
                    `Display name: ${displayName}\n` +
                    `Description: ${description}\n` +
                    (details ? `Details: ${details}\n` : '');
                const raw = await converse({
                    modelId: GENERATOR_MODEL_ID,
                    system: buildSystemPrompt(promptType, fewShot),
                    userText,
                });
                const xml = extractXml(raw);
                if (!xml) throw new Error('model returned no XML');
                prompts[promptType] = xml;
                send({ type: 'section', promptType, label: meta.label, chars: xml.length, ok: true });
            } catch (e) {
                failures.push(`${promptType}: ${e.message}`);
                prompts[promptType] = `<!-- ERROR generating ${promptType}: ${e.message} -->`;
                send({ type: 'section', promptType, label: meta.label, ok: false, message: e.message });
            }
        }

        if (aborted) return;

        // Every section failing means the Bedrock call itself is broken (creds,
        // region, model access) rather than one bad generation. Say so, instead
        // of handing back six comment stubs that look like content.
        if (failures.length === total) {
            return fail(`Prompt generation failed. ${failures[0]}`);
        }
        send({ type: 'done', specialistName, prompts, warnings: failures });
        res.end();
    });

    // ── Step 3: assemble the config for review ──

    app.post('/api/wizard/preview', async (req, res) => {
        try {
            const form = req.body || {};
            const specialistName = sanitizeName(form.displayName);
            if (!specialistName) return res.json({ error: 'Specialist name is required' });
            const exampleCount = Number(form.exampleCount) || 0;
            await assertModelsAvailable([form.primaryModel, form.fallback1, form.fallback2]);
            res.json({
                specialist_name: specialistName,
                registry_entry: {
                    name: specialistName,
                    display_name: form.displayName,
                    description: form.description,
                    enabled: true,
                    wizard_managed: true,
                },
                manifest: buildManifest({ ...form, specialistName, exampleCount }),
                schema: buildSchema({ ...form, specialistName }),
                files_to_write: [
                    'custom_specialists/specialist_registry.json',
                    `custom_specialists/manifests/${specialistName}.json`,
                    `custom_specialists/schemas/${specialistName}.json`,
                    ...PROMPT_TYPES.map(t => `custom_specialists/prompts/${specialistName}/${shortNameOf(specialistName)}_${t}.xml`),
                ],
            });
        } catch (e) {
            res.json({ error: e.message });
        }
    });

    // ── Step 4a: write the artifacts ──

    app.post('/api/wizard/save', async (req, res) => {
        const out = [];
        try {
            const form = req.body || {};
            const prompts = form.prompts || {};
            const specialistName = sanitizeName(form.displayName);
            if (!specialistName) return res.json({ error: 'Specialist name is required' });

            const missing = PROMPT_TYPES.filter(t => !String(prompts[t] || '').trim());
            if (missing.length) {
                return res.json({ error: `Missing prompts: ${missing.join(', ')}. Go back and generate them.` });
            }

            const short = shortNameOf(specialistName);
            const examples = Array.isArray(form.examples) ? form.examples.slice(0, 6) : [];

            for (const dir of ['manifests', 'schemas', `prompts/${specialistName}`]) {
                mkdirSync(safePath(dir), { recursive: true });
            }

            await assertModelsAvailable([form.primaryModel, form.fallback1, form.fallback2]);
            const manifest = buildManifest({ ...form, specialistName, exampleCount: examples.length });
            const manifestPath = safePath('manifests', `${specialistName}.json`);
            await writeFile(manifestPath, JSON.stringify(manifest, null, 4), 'utf-8');
            out.push(`✓ manifest  custom_specialists/manifests/${specialistName}.json`);

            const schemaPath = safePath('schemas', `${specialistName}.json`);
            await writeFile(schemaPath, JSON.stringify(buildSchema({ ...form, specialistName }), null, 2), 'utf-8');
            out.push(`✓ schema    custom_specialists/schemas/${specialistName}.json`);

            for (const promptType of PROMPT_TYPES) {
                const file = `${short}_${promptType}.xml`;
                await writeFile(safePath('prompts', specialistName, file), String(prompts[promptType]), 'utf-8');
            }
            out.push(`✓ prompts   custom_specialists/prompts/${specialistName}/ (${PROMPT_TYPES.length} files)`);

            // Example images arrive as base64 data URLs from the browser.
            if (examples.length) {
                const dir = safePath('prompts', specialistName, 'few-shot-images');
                mkdirSync(dir, { recursive: true });
                let written = 0;
                for (let i = 0; i < examples.length; i++) {
                    const item = examples[i] || {};
                    const b64 = String(item.data || '').split(',').pop();
                    if (!b64) continue;
                    let ext = extname(String(item.name || '')).toLowerCase();
                    if (!/^\.[a-z0-9]+$/.test(ext)) ext = '.png';
                    await writeFile(safePath('prompts', specialistName, 'few-shot-images', `example_${i + 1}${ext}`), Buffer.from(b64, 'base64'));
                    written++;
                }
                out.push(`✓ examples  ${written} image(s)`);
            }

            // Registry is shared across every custom specialist: read, replace
            // this one entry, write back. Never clobber other entries.
            const registryPath = safePath('specialist_registry.json');
            let registry = { specialists: [] };
            if (existsSync(registryPath)) {
                try {
                    const parsed = JSON.parse(await readFile(registryPath, 'utf-8'));
                    if (Array.isArray(parsed?.specialists)) registry = parsed;
                } catch {
                    out.push('⚠ existing registry was unreadable, starting a new one');
                }
            }
            registry.specialists = registry.specialists.filter(s => s?.name !== specialistName);
            registry.specialists.push({
                name: specialistName,
                display_name: form.displayName,
                description: form.description,
                enabled: true,
                created_at: new Date().toISOString(),
                wizard_managed: true,
            });
            await writeFile(registryPath, JSON.stringify(registry, null, 2), 'utf-8');
            out.push(`✓ registry  ${registry.specialists.length} custom specialist(s)`);

            out.push('', 'Saved. Click "Deploy Stack" to push this to AWS.');
            res.json({ specialistName, output: out.join('\n') });
        } catch (e) {
            out.push(`❌ ${e.message}`);
            res.json({ error: e.message, output: out.join('\n') });
        }
    });

    // ── Step 4b: deploy the stack ──
    //
    // CDK takes minutes, so this streams over SSE into the shared log panel
    // rather than blocking a plain POST until it times out.

    app.post('/api/wizard/deploy', (_req, res) => {
        const script = resolve(DEPLOY_DIR, 'deploy_custom_specialists.sh');
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.flushHeaders?.();

        const send = (obj) => { try { res.write(`data: ${JSON.stringify(obj)}\n\n`); } catch { } };

        if (!existsSync(script)) {
            send({ type: 'stderr', text: `Deploy script not found: ${script}\n` });
            send({ type: 'done', code: 1 });
            return res.end();
        }
        if (!existsSync(resolve(CUSTOM_DIR, 'specialist_registry.json'))) {
            send({ type: 'stderr', text: 'No specialist registry found. Save a specialist first.\n' });
            send({ type: 'done', code: 1 });
            return res.end();
        }

        send({ type: 'stdout', text: '▶ Running: deploy_custom_specialists.sh\n' });
        const proc = spawn('bash', [script], { cwd: DEPLOY_DIR, env: process.env });
        const heartbeat = setInterval(() => { try { res.write(': heartbeat\n\n'); } catch { } }, 15000);
        const cleanup = () => clearInterval(heartbeat);

        proc.stdout.on('data', d => send({ type: 'stdout', text: d.toString() }));
        proc.stderr.on('data', d => send({ type: 'stderr', text: d.toString() }));
        proc.on('error', (err) => {
            cleanup();
            send({ type: 'stderr', text: `Process error: ${err.message}\n` });
            send({ type: 'done', code: 1 });
            res.end();
        });
        proc.on('close', (code, signal) => {
            cleanup();
            send({ type: 'done', code: code ?? (signal ? 1 : 0) });
            res.end();
        });
        res.on('close', () => { cleanup(); if (!proc.killed) proc.kill(); });
    });
}
