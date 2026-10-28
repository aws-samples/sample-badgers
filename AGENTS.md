# Repository Guidelines

## Project Structure & Module Organization

BADGERS is a Python/CDK document-analysis application with a React UI.

- `deployment/` contains CDK stacks, profile resolution, deployment scripts,
  Lambda code, runtime code, and deployment tests.
- `deployment/lambdas/code/` holds ZIP-based Lambda specialists;
  `deployment/lambdas/containers/` holds container specialists.
- `deployment/s3_files/` contains versioned runtime configuration: schemas,
  manifests, prompts, workflows, routing policies, and contracts.
- `deployment/profiles/` defines Full, Headless, and Specialists products plus
  reusable specialist packs.
- `ui/` contains the Vite/React client and Express server.
- `deployment/tests/` contains Python tests; UI route tests live at
  `ui/server/routes/core.test.js`.
- `.archive/` is local/reference material and is intentionally ignored.

## Build, Test, and Development Commands

Run Python checks from `deployment/`:

```bash
PYTHONPATH="$PWD/.." uv run pytest -q tests
uv run cdk synth
```

Use profile-aware synthesis when validating a deployment change:

```bash
DEPLOYMENT_ID=dev STACK_SUFFIX=a1b BADGERS_PROFILE=headless uv run cdk synth
```

Run UI checks from `ui/`:

```bash
npm test
npm run build
npm run dev
```

Use `./deploy.sh` for guided deployments. Do not treat a local CDK synth as
authorization to deploy or push container images.

## Coding Style & Naming Conventions

Use Python 3.12, four-space indentation, type annotations for new public
functions, and `snake_case` names. Keep Lambda handlers thin and place parsing,
validation, and policy logic in testable helpers. Use `PascalCase` for CDK
construct classes. JavaScript/React follows the existing ES module style:
`camelCase` functions and `PascalCase` components.

## Testing & Configuration

Name Python tests `test_*.py`; add focused coverage for profile resolution,
release staging, routing, and error paths when changing deployments. Configuration
under `deployment/s3_files/` is release-pinned: update its matching schema,
manifest, prompts, and tests together. Never commit generated `.archive`,
`cdk.out`, layers, scan reports, credentials, or staged container dependencies.

## Commits & Pull Requests

Follow the established Conventional Commit pattern, for example
`feat(deploy): add profile-aware configuration` or `fix(security): harden XML parsing`.
Keep commits scoped. PRs should state the affected product profiles, deployment
or migration implications, validation commands run, and screenshots for UI changes.
