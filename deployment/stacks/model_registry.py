"""Model registry reader — the single authoring surface for BADGERS' model set.

The registry lives at ``deployment/s3_files/config/model_registry.json`` and is read
**at synth time** by the CDK stacks that create inference profiles and IAM statements.
It cannot live in the config bucket, because CDK needs it before that bucket exists.

Two lists, not one. This file is *intent*: what models the project knows about. The
deployed reality — model ID to application inference profile ARN — is written to SSM at
``/badgers-{deployment_id}/model-profiles`` by ``InferenceProfilesStack``, in the same
loop that creates the profiles. That co-location is deliberate: the parameter cannot name
a model that has no profile.

Validation runs on load, so a malformed registry fails at ``cdk synth`` rather than inside
a Lambda under load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Repo-relative default. `deployment/stacks/model_registry.py` -> `deployment/`
_DEFAULT_PATH = (
    Path(__file__).resolve().parent.parent
    / "s3_files"
    / "config"
    / "model_registry.json"
)

VALID_PROVIDERS = {"anthropic", "amazon", "openai", "moonshotai", "mistral", "google"}
VALID_TRANSPORTS = {"converse", "invoke", "mantle"}
VALID_THINKING = {None, "extended", "adaptive"}

#: Three states, differing in what AWS resources they cause and whether the wizard offers
#: them:
#:
#: ``active``    provisioned, granted, offered in the wizard.
#: ``retiring``  provisioned and granted but **not** offered, so manifests that still name
#:               them keep working through the release that stops offering them.
#: ``disabled``  **not** provisioned, **not** granted, **not** offered, absent from SSM.
#:               The entry stays in the file as documentation of a model this deployment
#:               has deliberately not adopted — one whose provider terms, price, or
#:               region footprint an account declines, for instance. Flipping one word
#:               re-enables it, which beats deleting the entry and later reconstructing
#:               its prices by hand.
VALID_STATUS = {"active", "retiring", "disabled"}

#: Statuses that cause AWS resources to exist. ``disabled`` is deliberately absent.
PROVISIONED_STATUS = {"active", "retiring"}


class RegistryError(Exception):
    """Raised when the model registry is missing, malformed, or self-inconsistent."""


def model_profiles_param_name(deployment_id: str) -> str:
    """SSM parameter holding the model ID -> profile ARN map for a deployment.

    Defined once here because four places need it: ``InferenceProfilesStack`` writes it, and
    the specialist, AgentCore runtime, and custom-specialist stacks pass it to their
    functions as ``MODEL_PROFILES_PARAM``. String-building the convention in each of them is
    how the per-model environment variables drifted in the first place.
    """
    return f"/badgers-{deployment_id}/model-profiles"


_GEO_PREFIXES = ("us.", "eu.", "jp.", "au.", "in.", "global.")


def foundation_model_id(geo_model_id: str) -> str:
    """Strip the geo prefix to get the base foundation model ID.

    ``us.anthropic.claude-sonnet-4-6`` -> ``anthropic.claude-sonnet-4-6``

    Used for ``foundation-model/*`` ARNs. AWS requires the underlying foundation model to
    be granted alongside the inference profile, even for models that offer no in-Region
    inference at all. A key that carries no geo prefix (an In-Region-only model, see
    ``is_cross_region``) is already a bare foundation-model ID and is returned unchanged.
    """
    for prefix in _GEO_PREFIXES:
        if geo_model_id.startswith(prefix):
            return geo_model_id[len(prefix) :]
    return geo_model_id


def is_cross_region(spec: dict[str, Any]) -> bool:
    """Whether the model is reached through a cross-Region system inference profile.

    ``True`` (the default, and every model until an In-Region-only entry is added): the
    registry key is a ``us.`` geo ID and the application profile wraps the system-defined
    cross-Region profile of the same name.

    ``False``: the model offers no cross-Region inference (a model card listing
    ``Geo: Not supported`` / ``Global: Not supported``), so there is no ``us.*`` system
    profile to wrap. The registry key is the bare foundation-model ID, the application
    profile wraps the foundation model directly in the deploy Region, and the system-profile
    IAM grant is skipped because no such profile exists.

    Not meaningful for mantle models (see ``is_mantle``), which have no inference profile of
    any kind; the validator rejects ``cross_region`` on a mantle entry.
    """
    return bool(spec.get("cross_region", True))


def is_mantle(spec: dict[str, Any]) -> bool:
    """Whether the model is reached through the OpenAI-compatible ``bedrock-mantle`` endpoint
    rather than Converse.

    Mantle models (e.g. Gemma 4 31B, whose model card lists ``bedrock-mantle`` as the only
    endpoint) have no application inference profile: profiles wrap Converse/InvokeModel
    routing, which mantle does not use. Their cost is attributed through Amazon Bedrock
    Projects — the account default project unless a project ID is passed — not a profile.
    So they are excluded from profile creation, the SSM profile map, and the two
    inference-profile IAM statements; they are granted the foundation model and the default
    project instead, and invoked over HTTP (SigV4) by the foundation layer.
    """
    return spec.get("transport") == "mantle"


def profile_slug(geo_model_id: str) -> str:
    """Build a short, stable, DNS-ish slug from a geo model ID.

    ``us.openai.gpt-5.6-terra``   -> ``gpt-5-6-terra``
    ``us.amazon.nova-2-lite-v1:0`` -> ``nova-2-lite``
    """
    base = foundation_model_id(geo_model_id)
    _, _, remainder = base.partition(".")
    slug = remainder or base
    for suffix in ("-v1:0", "-v1", ":0"):
        if slug.endswith(suffix):
            slug = slug[: -len(suffix)]
            break
    return slug.replace(".", "-").replace(":", "-")


def construct_id_for(geo_model_id: str) -> str:
    """CloudFormation construct ID for a model's profile."""
    parts = [p for p in profile_slug(geo_model_id).split("-") if p]
    return "".join(p.capitalize() for p in parts) + "Profile"


def profile_name_for(geo_model_id: str, deployment_id: str) -> str:
    """Application inference profile name, including the deployment suffix."""
    return f"badgers-{profile_slug(geo_model_id)}-{deployment_id}"


def output_id_for(geo_model_id: str) -> str:
    """CfnOutput logical ID for a model's profile ARN."""
    return construct_id_for(geo_model_id) + "Arn"


def _validate(models: dict[str, Any], source: Path) -> None:
    """Reject a registry that would fail later, in a place that is harder to debug."""
    if not models:
        raise RegistryError(f"{source}: 'models' is empty")

    for model_id, spec in models.items():
        where = f"{source}: model '{model_id}'"

        if not isinstance(spec, dict):
            raise RegistryError(f"{where}: entry must be an object")

        if spec.get("transport") == "mantle":
            # Mantle (OpenAI-compatible) models have no inference profile and no
            # cross-Region concept. The key is the bare model ID exactly as the model card
            # lists it for the bedrock-mantle endpoint (e.g. `google.gemma-4-31b`).
            if "cross_region" in spec:
                raise RegistryError(
                    f"{where}: 'cross_region' is meaningless for a mantle model — mantle "
                    f"has no inference profile to route"
                )
            if model_id.startswith(_GEO_PREFIXES):
                raise RegistryError(
                    f"{where}: a mantle model key must be the bare model ID, not "
                    f"geo-prefixed"
                )
        elif not isinstance(spec.get("cross_region", True), bool):
            raise RegistryError(f"{where}: 'cross_region' must be a boolean if present")

        elif is_cross_region(spec):
            # The default. The key names a `us.` geo system profile that the application
            # profile wraps; a bare or non-`us.` ID would silently bypass cross-Region
            # routing.
            if not model_id.startswith("us."):
                raise RegistryError(
                    f"{where}: must start with 'us.' (US geo). For a model with no "
                    f'cross-Region inference, set "cross_region": false and use the '
                    f"bare foundation-model ID"
                )
        else:
            # In-Region only. The key must be the bare foundation-model ID, invoked directly
            # and wrapped by an application profile over the foundation model — never
            # geo-prefixed, or the profile would point at a system profile that does not
            # exist.
            if model_id.startswith(_GEO_PREFIXES):
                raise RegistryError(
                    f"{where}: 'cross_region' is false, so the key must be the bare "
                    f"foundation-model ID, not geo-prefixed"
                )

        for field in ("display_name", "provider", "transport", "status"):
            if not spec.get(field):
                raise RegistryError(f"{where}: missing required field '{field}'")

        if spec["provider"] not in VALID_PROVIDERS:
            raise RegistryError(
                f"{where}: unknown provider {spec['provider']!r} "
                f"(expected one of {sorted(VALID_PROVIDERS)})"
            )
        if spec["transport"] not in VALID_TRANSPORTS:
            raise RegistryError(
                f"{where}: unknown transport {spec['transport']!r} "
                f"(expected one of {sorted(VALID_TRANSPORTS)})"
            )
        if spec["status"] not in VALID_STATUS:
            raise RegistryError(
                f"{where}: unknown status {spec['status']!r} "
                f"(expected one of {sorted(VALID_STATUS)})"
            )

        if spec["status"] != "active":
            continue

        # Fields required only of models the wizard will offer. A `retiring` model is
        # never displayed or priced, so it needs none of this.
        for field in ("price_in", "price_out"):
            if not isinstance(spec.get(field), (int, float)):
                raise RegistryError(f"{where}: active model needs numeric '{field}'")

        if "thinking" not in spec:
            raise RegistryError(
                f"{where}: active model must state 'thinking' explicitly "
                f"(null, 'extended', or 'adaptive') — absence is not 'unsupported'"
            )
        if spec["thinking"] not in VALID_THINKING:
            raise RegistryError(
                f"{where}: unknown thinking {spec['thinking']!r} "
                f"(expected null, 'extended', or 'adaptive')"
            )
        if not isinstance(spec.get("prompt_caching"), bool):
            raise RegistryError(
                f"{where}: active model needs boolean 'prompt_caching' "
                f"(availability on the configured transport, not the model in general)"
            )

        # Some models reason unless told not to -- Claude Opus 5's model card: "adaptive
        # thinking is on by default; can be disabled". A request that asks for no thinking
        # is still a thinking request on such a model, which matters because Claude
        # requires temperature 1 whenever thinking is on. The flag records that so the
        # runtime can apply the rule; it is meaningless on a model that cannot think.
        if spec.get("thinking_default_on") and spec["thinking"] is None:
            raise RegistryError(
                f"{where}: 'thinking_default_on' is meaningless when thinking is null"
            )
        if not isinstance(spec.get("thinking_default_on", False), bool):
            raise RegistryError(
                f"{where}: 'thinking_default_on' must be a boolean if present"
            )

        long_fields = ("price_in_long", "price_out_long", "long_context_threshold")
        present = [f for f in long_fields if f in spec]
        if present and len(present) != len(long_fields):
            raise RegistryError(
                f"{where}: long-context pricing is partial — has {present}, "
                f"needs all of {list(long_fields)}"
            )


def load_registry(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Load and validate the registry, returning ``{model_id: spec}``.

    Raises ``RegistryError`` on anything that would otherwise surface at deploy time or,
    worse, at runtime.
    """
    source = Path(path) if path else _DEFAULT_PATH

    if not source.exists():
        raise RegistryError(
            f"Model registry not found at {source}. "
            f"It is the source of truth for profiles and IAM and cannot be defaulted."
        )

    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"{source}: invalid JSON — {exc}") from exc

    if not isinstance(raw, dict) or "models" not in raw:
        raise RegistryError(f"{source}: expected an object with a 'models' key")

    models = raw["models"]
    if not isinstance(models, dict):
        raise RegistryError(f"{source}: 'models' must be an object")

    _validate(models, source)
    return models


def active_models(models: dict[str, Any]) -> dict[str, Any]:
    """Models offered in the wizard. Subset of what gets provisioned."""
    return {k: v for k, v in models.items() if v["status"] == "active"}


def retiring_models(models: dict[str, Any]) -> dict[str, Any]:
    """Models still provisioned and invocable, but no longer offered for new work."""
    return {k: v for k, v in models.items() if v["status"] == "retiring"}


def provisioned_models(models: dict[str, Any]) -> dict[str, Any]:
    """Models that get a profile, IAM grants, and an SSM entry.

    This — not the full registry — is what the CDK stacks iterate. ``disabled`` entries are
    documentation only and must produce no AWS resources, or declining a model would still
    leave its profile and grants behind.
    """
    return {k: v for k, v in models.items() if v["status"] in PROVISIONED_STATUS}


def disabled_models(models: dict[str, Any]) -> dict[str, Any]:
    """Models present in the file but deliberately not adopted by this deployment."""
    return {k: v for k, v in models.items() if v["status"] == "disabled"}


def has_provider(models: dict[str, Any], provider: str) -> bool:
    """Whether any model comes from the given provider.

    Used to gate provider-specific IAM: OpenAI models additionally require
    ``bedrock:InvokeModel`` on the account's default project.
    """
    return any(spec["provider"] == provider for spec in models.values())
