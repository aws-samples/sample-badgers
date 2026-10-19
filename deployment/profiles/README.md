# BADGERS deployment profiles

Profiles are declarative deployment inputs. They resolve a named product
offering into an explicit specialist set and capability map before CDK or AWS
resource creation.

On `release/6.0.0-beta`, profiles are part of a private release candidate. The
stable customer source release is `v5.0.2`; the beta branch has no public
release tag. See the repository [Release Guide](../../RELEASE_GUIDE.md) for
the source-release and publication policy.

```bash
./.venv/bin/python deployment/profile_resolver.py full
./.venv/bin/python deployment/profile_resolver.py headless

# Guided menu for the Headless product
./deploy.sh headless

# Non-interactive profile selection; option 9 runs all applicable steps
./deploy.sh headless 9

# Direct Lambda specialist pack; no agent runtime or Gateway
./deploy.sh specialists --pack tables-and-forms 9
```

`full.json` intentionally selects every specialist enabled by the existing
`deployment_config.json`, preserving today’s deployment behavior.

`headless.json` selects curated specialist packs and is wired into the
deployment engine. Headless synthesizes and deploys its selected specialists,
agent runtime, Gateway, Cognito machine identity, storage, model profiles, and
the minimal DynamoDB operation ledger and private workflow runner. It omits
ECS, VPC, UI image build, specialist wizard, session memory, X-Ray Transaction
Search, and WebSocket handler registration. The initial Headless workflow is
limited to the curated PDF/image capabilities; native DOCX/PPTX support is not
implicitly enabled by the Full profile's specialist set.

Headless stages only the shared agent/prompt core and the selected
specialists' schemas, manifests, and prompts before uploading configuration.
It does not sync the entire specialist catalog to its configuration bucket.
Each deployment ID is bound to one product profile in `.deploy-state`; use a
new ID rather than attempting to convert an existing Full deployment in place.

Every complete Step 3 upload builds a deterministic `release-manifest.json`,
hashes its profile resolution and staged file contents, uploads the result
under `releases/<sha256>/`, and stores the SHA in deployment state. Later CDK
steps pass that release prefix to specialist functions, Gateway schemas, and
the agent runtime. Re-run dependent deployment steps after a configuration
release changes so their pinned prefix advances together.

This is a **configuration release**, not a Git release: it is private to one
deployment and pins runtime configuration. A customer source release is an
immutable Git tag such as `v5.0.2`; the future 6.0 beta tag must be explicitly
approved before publication. See [Release terminology](../../RELEASE_GUIDE.md#release-terminology).

Specialist packs live under `specialist_packs/`. A pack is a named, ordered
selection of existing specialists. The resolver validates that each selected
specialist exists and is enabled in the current deployment configuration.

`specialists.json` is user-selectable: omit `--pack` for Document Essentials
or repeat `--pack` to compose a direct-use installation. The resolved pack
selection is written to deployment state and cannot be changed in place.
