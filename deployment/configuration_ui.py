#!/usr/bin/env python3
"""Loopback-only local UI for planning BADGERS deployment configuration.

The UI has no AWS credentials, AWS SDK calls, shell execution, or deployment
endpoint. It edits and validates desired state through the same selection and
routing modules used by deploy.sh. Applying a reviewed plan remains a CLI step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    from . import observability_configuration, routing_configuration, specialist_selection
except ImportError:
    import observability_configuration
    import routing_configuration
    import specialist_selection


_DEPLOYMENT_ID = re.compile(r"^[a-z][a-z0-9-]{0,15}$")
_PROFILES = frozenset({"full", "headless", "specialists"})
_MAX_BODY_BYTES = 64 * 1024
_STAGE_KEYS = (
    "layers_built",
    "infra_complete",
    "s3_files_uploaded",
    "specialists_complete",
    "gateway_complete",
    "runtime_image_pushed",
    "runtime_deployed",
    "ui_image_pushed",
    "ecs_deployed",
)


class UIError(ValueError):
    """A safe, user-facing local configuration error."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _state_path(repo_root: Path, deployment_id: str) -> Path:
    if not _DEPLOYMENT_ID.fullmatch(deployment_id):
        raise UIError("Deployment IDs use lowercase letters, digits, and hyphens only.")
    return repo_root / ".deploy-state" / f"{deployment_id}.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise UIError(f"Deployment state does not exist: {path.stem}") from error
    except json.JSONDecodeError as error:
        raise UIError(f"Deployment state is invalid: {error}") from error
    if not isinstance(value, dict):
        raise UIError("Deployment state must be a JSON object.")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _initial_state(deployment_id: str, profile: str) -> dict[str, Any]:
    return {
        "deployment_id": deployment_id,
        "started_at": _now(),
        "product_profile": profile,
        **{key: False for key in _STAGE_KEYS},
        "gateway_url": "",
    }


def _ensure_deployment(repo_root: Path, deployment_id: str, profile: str) -> Path:
    if profile not in _PROFILES:
        raise UIError("Choose Full, Headless, or Specialists.")
    path = _state_path(repo_root, deployment_id)
    if path.exists():
        state = _read_json(path)
        current = state.get("product_profile")
        if current and current != profile:
            raise UIError(
                f"Deployment '{deployment_id}' belongs to the {current} product profile."
            )
    else:
        _write_json(path, _initial_state(deployment_id, profile))
    specialist_selection.load_or_create_desired(path, profile)
    observability_configuration.load_or_create_desired(path, profile)
    try:
        routing_configuration.ensure(path, profile)
    except routing_configuration.RoutingConfigurationError:
        # A route can become invalid only after its specialist prerequisite was
        # removed. Preserve it for review instead of replacing it silently.
        pass
    return path


def _invalidate_consumers(path: Path, *, infrastructure: bool = False) -> None:
    state = _read_json(path)
    keys = [
        "s3_files_uploaded",
        "specialists_complete",
        "gateway_complete",
        "runtime_image_pushed",
        "runtime_deployed",
    ]
    if infrastructure:
        keys.insert(0, "infra_complete")
    for key in keys:
        state[key] = False
        state.pop(f"{key}_at", None)
    _write_json(path, state)


def _routing_summary(state: dict[str, Any]) -> dict[str, Any]:
    desired = state.get("desired_routing_configuration")
    applied = state.get("applied_routing_configuration")
    if not isinstance(desired, dict):
        return {"templates": [], "in_sync": False, "error": "Routing is not configured."}
    return {
        "templates": desired.get("templates", []),
        "fingerprint": desired.get("fingerprint", ""),
        "revision": state.get("routing_configuration_revision", 0),
        "in_sync": isinstance(applied, dict)
        and applied.get("fingerprint") == desired.get("fingerprint"),
    }


def configuration(repo_root: Path, deployment_id: str, profile: str) -> dict[str, Any]:
    path = _ensure_deployment(repo_root, deployment_id, profile)
    desired, _ = specialist_selection.load_or_create_desired(path, profile)
    selection_plan = specialist_selection.reconciliation(path, desired)
    try:
        routing_configuration.ensure(path, profile)
        state = _read_json(path)
        routing = _routing_summary(state)
    except routing_configuration.RoutingConfigurationError as error:
        state = _read_json(path)
        routing = {**_routing_summary(state), "error": str(error)}
    observability = observability_configuration.summary(path, profile, desired)
    state = _read_json(path)
    return {
        "deployment_id": deployment_id,
        "profile": profile,
        "selection": desired,
        "selection_plan": selection_plan,
        "routing": routing,
        "observability": observability,
        "state": {
            "specialist_revision": state.get("specialist_selection_revision", 0),
            "routing_revision": state.get("routing_configuration_revision", 0),
            "observability_revision": state.get("observability_configuration_revision", 0),
        },
    }


def configurations(repo_root: Path) -> list[dict[str, str]]:
    """Return valid local deployment configuration records for the landing screen."""

    state_dir = repo_root / ".deploy-state"
    if not state_dir.is_dir():
        return []
    records: list[dict[str, str]] = []
    for path in sorted(state_dir.glob("*.json")):
        try:
            state = _read_json(path)
            deployment_id = str(state.get("deployment_id") or "")
            profile = str(state.get("product_profile") or "")
            if (
                not _DEPLOYMENT_ID.fullmatch(deployment_id)
                or path != _state_path(repo_root, deployment_id)
                or profile not in _PROFILES
            ):
                continue
        except UIError:
            continue
        records.append(
            {
                "deployment_id": deployment_id,
                "profile": profile,
                "started_at": str(state.get("started_at") or ""),
            }
        )
    return records


def delete_configuration(repo_root: Path, payload: dict[str, Any]) -> dict[str, str]:
    """Delete one local deployment desired-state record after exact confirmation."""

    deployment_id = str(payload.get("deployment_id") or "")
    if payload.get("confirmation") != deployment_id:
        raise UIError("Type the exact deployment ID to delete its local configuration.")
    path = _state_path(repo_root, deployment_id)
    _read_json(path)
    path.unlink()
    return {"deleted": deployment_id}


def _list_of_strings(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise UIError(f"'{label}' must be a list of non-empty strings.")
    return [item.strip() for item in value]


def save_selection(repo_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    deployment_id = str(payload.get("deployment_id") or "")
    profile = str(payload.get("profile") or "")
    path = _ensure_deployment(repo_root, deployment_id, profile)
    state = _read_json(path)
    expected = payload.get("expected_revision")
    if expected is not None and expected != state.get("specialist_selection_revision", 0):
        raise UIError("Specialist selection changed in another session; reload before saving.")
    packs = _list_of_strings(payload.get("packs", []), "packs")
    specialists = _list_of_strings(payload.get("specialists", []), "specialists")
    desired, changed = specialist_selection.load_or_create_desired(
        path, profile, packs=packs, direct_specialists=specialists
    )
    if changed:
        _invalidate_consumers(path, infrastructure=True)
    return {**configuration(repo_root, deployment_id, profile), "changed": changed}


def save_routing(repo_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    deployment_id = str(payload.get("deployment_id") or "")
    profile = str(payload.get("profile") or "")
    path = _ensure_deployment(repo_root, deployment_id, profile)
    state = _read_json(path)
    expected = payload.get("expected_revision")
    if expected is not None and expected != state.get("routing_configuration_revision", 0):
        raise UIError("Workflow routing changed in another session; reload before saving.")
    templates = _list_of_strings(payload.get("templates", []), "templates")
    _, changed = routing_configuration.save_templates(path, profile, templates)
    if changed:
        _invalidate_consumers(path)
    return {**configuration(repo_root, deployment_id, profile), "changed": changed}


def save_observability(repo_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Persist a local monitoring choice without making an AWS call."""

    deployment_id = str(payload.get("deployment_id") or "")
    profile = str(payload.get("profile") or "")
    path = _ensure_deployment(repo_root, deployment_id, profile)
    state = _read_json(path)
    expected = payload.get("expected_revision")
    if expected is not None and expected != state.get("observability_configuration_revision", 0):
        raise UIError(
            "Observability configuration changed in another session; reload before saving."
        )
    try:
        _, changed = observability_configuration.save(
            path,
            profile,
            mode=str(payload.get("mode") or ""),
            notification_kind=str(payload.get("notification_kind") or ""),
            notification_target=str(payload.get("notification_target") or ""),
        )
    except observability_configuration.ObservabilityConfigurationError as error:
        raise UIError(str(error)) from error
    return {**configuration(repo_root, deployment_id, profile), "changed": changed}


def plan(repo_root: Path, deployment_id: str, profile: str) -> dict[str, Any]:
    current = configuration(repo_root, deployment_id, profile)
    routing = current["routing"]
    if routing.get("error"):
        raise UIError(str(routing["error"]))
    selection_plan = current["selection_plan"]
    state = _read_json(_state_path(repo_root, deployment_id))
    applied_routing = state.get("applied_routing_configuration")
    routing_changed = not (
        isinstance(applied_routing, dict)
        and applied_routing.get("fingerprint") == routing.get("fingerprint")
    )
    stages = list(selection_plan["required_stages"])
    if routing_changed and "publish configuration" not in stages:
        stages.insert(0, "publish configuration")
    if routing_changed and profile != "specialists":
        stages.extend(
            stage
            for stage in ("reconcile Gateway targets", "redeploy Runtime with the new configuration release")
            if stage not in stages
        )
    return {
        **current,
        "routing_changed": routing_changed,
        "required_stages": stages,
        "observability_handoff": (
            "Observability is saved as a separate local add-on plan. Its future CLI "
            "deployer must check live account inventory, quotas, and notification "
            "permissions before it creates monitoring resources."
        ),
        "cli_next_step": f"./deploy.sh {profile} resume",
        "apply_note": (
            "Run the CLI command, select this deployment ID, and confirm there. "
            "The CLI re-runs preflight and validates this desired state before AWS changes."
        ),
    }


def packs(repo_root: Path, profile: str | None = None) -> list[dict[str, Any]]:
    del repo_root
    return specialist_selection.pack_definitions(profile)


def catalog(repo_root: Path) -> list[dict[str, str | bool]]:
    del repo_root
    return [
        {
            "name": name,
            "category": str(value.get("category", "Specialist")),
            "container": name in specialist_selection.CONTAINER_SPECIALISTS,
        }
        for name, value in sorted(specialist_selection._catalog().items())
        if value.get("enabled", True)
    ]


def save_pack(repo_root: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    del repo_root
    name = str(payload.get("name") or "")
    description = str(payload.get("description") or "")
    members = _list_of_strings(payload.get("specialists", []), "specialists")
    path = specialist_selection.PACKS_DIR / f"{name}.json"
    version = 1
    if path.is_file():
        version = int(json.loads(path.read_text(encoding="utf-8")).get("version", 0)) + 1
    specialist_selection.save_pack(name, description, members, version=version)
    return specialist_selection.pack_definitions()


def delete_pack(repo_root: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    name = str(payload.get("name") or "")
    if payload.get("confirmation") != name:
        raise UIError("Type the exact pack name to delete it.")
    specialist_selection.delete_pack(name, repo_root / ".deploy-state", confirmed=True)
    return specialist_selection.pack_definitions()


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BADGERS local configuration</title><style>
:root{--i:#202431;--m:#687083;--b:#f5f6fa;--p:#fff;--l:#e0e4ed;--v:#6244d6;--vs:#f0edff;--g:#16785b;--gs:#e9f8f2;--cg:#addfc4;--cg-border:#8bc9a7;--a:#a25d00;--as:#fff4df;font:14px/1.45 Inter,system-ui,sans-serif;color:var(--i);background:var(--b)}*{box-sizing:border-box}body{margin:0}.bar{height:58px;background:#211a3e;color:#fff;display:flex;align-items:center;justify-content:space-between;padding:0 28px}.brand{font-size:13px;font-weight:800;letter-spacing:.11em}.brand span{font-weight:400;letter-spacing:0;color:#d3cdef;margin-left:9px}.shell{display:grid;grid-template-columns:220px 1fr;min-height:calc(100vh - 58px)}nav{padding:23px 13px;background:#fcfcfe;border-right:1px solid var(--l)}.nav{width:100%;padding:10px 11px;text-align:left;border:1px solid transparent;border-radius:8px;background:none;color:var(--m);cursor:pointer}.nav:hover{background:#fff}.nav.on{background:var(--vs);border-color:#dad1ff;color:#4c35ae;font-weight:750}.note{font-size:11px;color:var(--m);line-height:1.6;padding:17px 11px;border-top:1px solid var(--l);margin-top:15px}main{max-width:1240px;margin:auto;width:100%;padding:32px}.view{display:none}.view.on{display:block}.ey{color:var(--v);font-size:10px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;margin:0 0 5px}h1{margin:0;font-size:26px;letter-spacing:-.035em}h2{margin:0;font-size:16px}.sub{max-width:730px;color:var(--m);margin:6px 0 22px}.head{display:flex;align-items:start;justify-content:space-between;gap:15px}.grid{display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:19px;align-items:start}.card{background:var(--p);border:1px solid var(--l);border-radius:11px;overflow:hidden;box-shadow:0 7px 23px #252b3b0b}.pad{padding:18px}.cardhead{padding:16px 18px;border-bottom:1px solid var(--l)}.cardhead p{margin:3px 0 0;color:var(--m);font-size:12px}.row{display:flex;align-items:center;gap:11px;padding:12px 18px;border-bottom:1px solid #eff1f5}.row:last-child{border:0}.row.select{cursor:pointer}.row.select:hover{background:#fbfaff}.row.on{background:#faf8ff}.check{width:20px;height:20px;display:grid;place-items:center;flex:none;border:1px solid #b8bfcc;border-radius:5px;background:#fff;color:#fff;font-weight:900}.on .check{border-color:var(--cg-border);background:var(--cg)}.copy{flex:1}.copy b,.copy small{display:block}.copy small{color:var(--m);font-size:11px}.pill{font-size:10px;font-weight:750;border:1px solid #d6dbe5;border-radius:999px;padding:3px 7px;color:#596174;background:#fbfcfd}.pill.g{color:var(--g);background:var(--gs);border-color:#bee7d8}.pill.a{color:var(--a);background:var(--as);border-color:#eed4a1}.btn{border:1px solid #cad1dd;border-radius:7px;padding:8px 10px;background:#fff;font-size:12px;font-weight:700;cursor:pointer}.btn.primary{background:var(--v);border-color:#5036c2;color:#fff}.btn.soft{background:var(--vs);border-color:#d9d1ff;color:#4d35b3}.side{padding:18px;background:linear-gradient(145deg,#35246e,#5e42cb);color:#fff}.side p{color:#e5e1fc;font-size:12px}.stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:15px 0}.stat{border:1px solid #ffffff33;border-radius:8px;padding:9px;background:#ffffff14}.stat b,.stat small{display:block}.stat b{font-size:20px}.stat small{font-size:10px;color:#e7e2fc}.detail{padding:16px 18px;border-top:1px solid var(--l)}.detail h3{margin:0 0 5px;font-size:12px}.detail p{margin:0;color:var(--m);font-size:12px}.flow{margin:10px 0 0 31px;display:flex;flex-wrap:wrap;gap:5px;align-items:center}.node{font-size:10px;font-weight:700;padding:4px 7px;border:1px solid #d9ddea;border-radius:5px;background:#fafbfc;color:#535c6e}.node.native{border-color:#d9d0ff;background:var(--vs);color:#5138b8}.arrow{color:#9aa2b2}.warn{margin:14px 18px;padding:10px;border:1px solid #f0d7a3;border-radius:8px;background:var(--as);color:#754500;font-size:12px}.hidden{display:none}.code{margin:10px 0;padding:9px;border-radius:6px;background:#17132b;color:#e9e4ff;font:12px ui-monospace,monospace}.modal{position:fixed;inset:0;background:#17132977;display:none;align-items:center;justify-content:center;padding:20px}.modal.on{display:flex}.dialog{width:min(560px,100%);background:#fff;border-radius:11px;box-shadow:0 22px 70px #0005;padding:20px}.field{margin:11px 0}.field label{display:block;font-size:11px;font-weight:800;margin-bottom:4px}.field input,.field select{width:100%;padding:9px;border:1px solid #cbd2df;border-radius:6px}.toast{position:fixed;right:22px;bottom:22px;display:none;max-width:390px;color:#fff;background:#25765d;border-radius:8px;padding:12px}.toast.on{display:block}@media(max-width:760px){.shell{grid-template-columns:1fr}nav{display:flex;overflow:auto;padding:9px;border-bottom:1px solid var(--l)}.nav{width:auto;white-space:nowrap}.note{display:none}main{padding:21px}.grid{grid-template-columns:1fr}.bar{padding:0 16px}.bar span{display:none}}
</style></head><body><header class="bar"><div class="brand">BADGERS <span>local configuration</span></div><small id="identity">loopback only · no AWS actions</small></header><div class="shell"><nav><button class="nav on" data-v="overview">Overview</button><button class="nav" data-v="specialists">Specialists</button><button class="nav" data-v="routing">Workflow routing</button><button class="nav" data-v="packs">Pack library</button><button class="nav" data-v="review">Review &amp; publish</button><p class="note">This local console saves desired state and creates a review plan. The normal CLI performs preflight and deployment.</p></nav><main>
<section id="overview" class="view on"><p class="ey">Deployment configuration</p><div class="head"><div><h1>Configure before deploying</h1><p class="sub">A local planning surface for packs, specialists, and reviewed workflow templates. AWS changes remain in the CLI.</p></div><button class="btn primary" data-to="review">Review update</button></div><div class="grid"><div class="card"><div class="cardhead"><h2>Choose a deployment</h2><p>New configuration creates only local state; it creates no AWS resources.</p></div><div class="pad"><div class="field"><label>Deployment ID</label><input id="deployment" value="beta"></div><div class="field"><label>Product profile</label><select id="profile"><option value="headless">Headless agent</option><option value="full">Full experience</option><option value="specialists">Specialists only</option></select></div><button class="btn primary" id="load">Load configuration</button></div></div><aside class="card"><div class="side"><p class="ey" style="color:#ddd6ff">Desired state</p><h2 id="title-id">Not loaded</h2><p id="summary">Choose a deployment to begin.</p><div class="stats"><div class="stat"><b id="s-count">—</b><small>specialists</small></div><div class="stat"><b id="w-count">—</b><small>workflows</small></div></div></div><div class="detail"><h3>Safety boundary</h3><p>No API here deploys, runs shell commands, or accesses AWS credentials.</p></div></aside></div></section>
<section id="specialists" class="view"><p class="ey">Capabilities</p><div class="head"><div><h1>Specialist selection</h1><p class="sub">Select packs plus explicit specialists. Any container requirement is automatically added to the reconciliation plan.</p></div><div><button class="btn" id="select-all-specialists">Select all</button> <button class="btn" id="deselect-all-specialists">Deselect all</button> <button class="btn primary" id="save-selection">Save selection</button></div></div><div class="grid"><div class="card"><div class="cardhead"><h2>Available selections</h2><p>Use the same desired-state model as the CLI picker.</p></div><div id="specialist-list"></div></div><aside class="card"><div class="side"><p class="ey" style="color:#ddd6ff">Resolved</p><h2 id="resolved">— specialists</h2><p id="container">Select a deployment first.</p></div><div class="detail"><h3>Dependency check</h3><p>Removing a specialist needed by a workflow makes routing invalid until it is updated. Nothing is silently removed in AWS.</p></div></aside></div></section>
<section id="routing" class="view"><p class="ey">Approved execution templates</p><div class="head"><div><h1>Workflow routing</h1><p class="sub">A pack makes specialists available; a template explicitly determines execution order and source-type routing.</p></div><div><button class="btn" id="select-all-routing">Select available</button> <button class="btn" id="deselect-all-routing">Deselect all</button> <button class="btn primary" id="save-routing">Save routing</button></div></div><div class="grid"><div class="card"><div class="cardhead"><h2>Workflow templates</h2><p>Native templates become available only when their processor is selected.</p></div><div id="routing-list"></div><div id="routing-error" class="warn hidden"></div></div><aside class="card"><div class="detail"><h3>Automatic dispatch</h3><p>An omitted workflow ID is selected by configured source type. An explicit request workflow ID always wins.</p></div><div class="detail"><h3>Guarded customization</h3><p>Templates are validated and release-pinned. Arbitrary graph editing is intentionally excluded.</p></div></aside></div></section>
<section id="packs" class="view"><p class="ey">Reusable definitions</p><div class="head"><div><h1>Pack library</h1><p class="sub">Packs are reusable selection presets. Editing one does not deploy it; the next configuration load surfaces its impact.</p></div><button class="btn primary" id="new-pack">Create pack</button></div><div class="grid"><div class="card"><div class="cardhead"><h2>Pack definitions</h2><p>A pack in use cannot be deleted until dependent deployments are migrated.</p></div><div id="pack-list"></div></div><aside class="card"><div class="detail"><h3>Order is not execution</h3><p>Pack membership is a composition convenience. Workflow templates define run order and artifact contracts.</p></div></aside></div></section>
<section id="review" class="view"><p class="ey">Reconciliation preview</p><div class="head"><div><h1>Review before the CLI applies</h1><p class="sub">The configuration service creates this local plan. The CLI re-validates it and runs preflight before any AWS mutation.</p></div><button class="btn soft" data-to="routing">Back to routing</button></div><div class="grid"><div class="card"><div class="cardhead"><h2>Required update path</h2><p id="plan-summary">Load a deployment to calculate the plan.</p></div><div id="plan-list"></div></div><aside class="card"><div class="side"><p class="ey" style="color:#ddd6ff">CLI executor</p><h2>Ready when you are</h2><p>The browser does not invoke deployment commands.</p><div class="code" id="cli-command">./deploy.sh headless resume</div><button class="btn" id="copy">Copy CLI command</button></div><div class="detail"><h3>Apply safely</h3><p id="apply-note">The CLI will ask you to select the deployment and confirm after fresh preflight.</p></div></aside></div></section>
</main></div><div id="pack-modal" class="modal"><div class="dialog"><div class="head"><div><p class="ey">Pack library</p><h2>Create or update pack</h2></div><button class="btn" data-close>Close</button></div><div class="field"><label>Name</label><input id="pack-name" placeholder="accessibility-review"></div><div class="field"><label>Description</label><input id="pack-description" placeholder="Short description"></div><div class="field"><label>Members (comma-separated specialist IDs)</label><input id="pack-members" placeholder="full_text_specialist, layout_specialist"></div><button class="btn primary" id="save-pack">Save pack</button></div></div><div id="toast" class="toast"></div>
<script>
const token=new URLSearchParams(location.search).get('token');let state=null,catalogEntries=[],packDefs=[],editingPack=null;
const requestedView=new URLSearchParams(location.search).get('view');
const api=(path,opt={})=>fetch(path,{...opt,headers:{'Content-Type':'application/json','X-Badgers-Session':token,...(opt.headers||{})}}).then(async r=>{let d=await r.json();if(!r.ok)throw Error(d.error||'Request failed');return d});
const toast=t=>{let e=document.querySelector('#toast');e.textContent=t;e.classList.add('on');setTimeout(()=>e.classList.remove('on'),3800)};
const esc=v=>String(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function switchTo(id){document.querySelectorAll('.view').forEach(x=>x.classList.toggle('on',x.id===id));document.querySelectorAll('.nav').forEach(x=>x.classList.toggle('on',x.dataset.v===id))}
document.querySelectorAll('.nav').forEach(x=>x.onclick=()=>switchTo(x.dataset.v));document.querySelectorAll('[data-to]').forEach(x=>x.onclick=()=>switchTo(x.dataset.to));
function row(label,sub,on,tag,attrs=''){return `<div class="row select ${on?'on':''}" ${attrs}><button class="check">${on?'✓':''}</button><div class="copy"><b>${esc(label)}</b><small>${esc(sub)}</small></div>${tag?`<span class="pill ${esc(tag.kind||'')}">${esc(tag.text)}</span>`:''}</div>`}
function render(){if(!state)return;let s=state.selection,route=state.routing;document.querySelector('#title-id').textContent=`${state.deployment_id} · ${state.profile}`;document.querySelector('#summary').textContent=route.error?'Routing needs attention':'Desired state ready for review';document.querySelector('#s-count').textContent=s.resolved_specialists.length;document.querySelector('#w-count').textContent=route.templates.length;document.querySelector('#resolved').textContent=`${s.resolved_specialists.length} specialists`;document.querySelector('#container').textContent=s.container_specialists.length?`Container builds: ${s.container_specialists.join(', ')}`:'ZIP Lambda selections only.';
let packRows=packDefs.map(p=>row(p.display_name,p.description,s.packs.includes(p.name),{text:`${p.specialists.length} specialists`,kind:'purple'},`data-pack="${esc(p.name)}"`)).join('');let directRows=catalogEntries.map(x=>row(x.name.replaceAll('_',' '),x.category,s.direct_specialists.includes(x.name),{text:x.container?'container required':'ZIP Lambda',kind:x.container?'a':''},`data-specialist="${esc(x.name)}"`)).join('');document.querySelector('#specialist-list').innerHTML='<div class="cardhead"><h2>Packs</h2></div>'+packRows+'<div class="cardhead"><h2>Individual specialists</h2></div>'+directRows;
document.querySelectorAll('#specialist-list .row').forEach(x=>x.onclick=()=>{x.classList.toggle('on');x.querySelector('.check').textContent=x.classList.contains('on')?'✓':''});
let templates=[['standard-document-analysis','PDF / image analysis','PDF, PNG, JPEG → page conversion → classify → approved page specialists','PDF/image',[],null],['native-docx-analysis','Native DOCX analysis','DOCX → native processor → private result artifact','DOCX',['docx_processor'],null],['native-pptx-analysis','Native PPTX analysis','PPTX → native processor → private result artifact','PPTX',['pptx_processor'],null],['native-xlsx-analysis','Native XLSX workbook analysis','XLSX → native workbook evidence → conditional pasted-image Vision enrichment → canonical BADGERS XML','XLSX',['xlsx_processor'],'headless']];document.querySelector('#routing-list').innerHTML=templates.map(t=>{let profileAvailable=!t[5]||s.profile===t[5];let enabled=profileAvailable&&t[4].every(name=>s.resolved_specialists.includes(name));let active=route.templates.includes(t[0]);let unavailable=!profileAvailable?'available in Headless only':t[0]==='native-xlsx-analysis'?'select the Workbook Analysis pack first':'select processor first';return row(t[1],t[2],active,{text:enabled?t[3]:unavailable,kind:enabled?'g':'a'},`data-template="${t[0]}" data-enabled="${enabled}"`)}).join('');
document.querySelectorAll('#routing-list .row').forEach(x=>x.onclick=()=>{if(x.dataset.enabled==='false')return;x.classList.toggle('on');x.querySelector('.check').textContent=x.classList.contains('on')?'✓':''});let err=document.querySelector('#routing-error');err.textContent=route.error||'';err.classList.toggle('hidden',!route.error);renderPlan()}
function renderPlan(){if(!state)return;api(`/api/plan?id=${encodeURIComponent(state.deployment_id)}&profile=${state.profile}`).then(p=>{document.querySelector('#plan-summary').textContent=`${p.required_stages.length} controlled stages are required before desired and applied state match.`;document.querySelector('#plan-list').innerHTML=p.required_stages.map((x,i)=>`<div class="row"><span class="pill purple">${i+1}</span><div class="copy"><b>${x}</b><small>${i===0?'Validated by a fresh preflight before AWS changes.':'Runs only if the reviewed configuration requires it.'}</small></div></div>`).join('');document.querySelector('#cli-command').textContent=p.cli_next_step;document.querySelector('#apply-note').textContent=p.apply_note}).catch(e=>toast(e.message))}
async function load(){try{let data=await Promise.all([api(`/api/configuration?id=${encodeURIComponent(document.querySelector('#deployment').value)}&profile=${document.querySelector('#profile').value}`),loadPacks(),loadCatalog()]);state=data[0];render();toast('Local desired state loaded.')}catch(e){toast(e.message)}}
async function loadPacks(){let profile=state?state.profile:document.querySelector('#profile').value;packDefs=await api(`/api/packs?profile=${encodeURIComponent(profile)}`);document.querySelector('#pack-list').innerHTML=packDefs.map(x=>`<div class="row"><div class="copy"><b>${esc(x.display_name)}</b><small>${x.specialists.length} members · version ${esc(x.version)} · ${esc(x.description)}</small></div><button class="btn" data-edit="${esc(x.name)}">Edit</button><button class="btn" data-delete="${esc(x.name)}">Delete</button></div>`).join('');document.querySelectorAll('[data-edit]').forEach(x=>x.onclick=()=>openPack(x.dataset.edit));document.querySelectorAll('[data-delete]').forEach(x=>x.onclick=()=>deletePack(x.dataset.delete))}
async function loadCatalog(){catalogEntries=await api('/api/catalog')}
document.querySelector('#load').onclick=load;document.querySelector('#save-selection').onclick=async()=>{if(!state)return toast('Load a deployment first.');try{let packs=[...document.querySelectorAll('[data-pack].on')].map(x=>x.dataset.pack),specialists=[...document.querySelectorAll('[data-specialist].on')].map(x=>x.dataset.specialist);state=await api('/api/selection',{method:'POST',body:JSON.stringify({deployment_id:state.deployment_id,profile:state.profile,packs,specialists,expected_revision:state.state.specialist_revision})});render();toast('Specialist desired state saved; dependent stages are pending.')}catch(e){toast(e.message)}};
document.querySelector('#save-routing').onclick=async()=>{if(!state)return toast('Load a deployment first.');try{let templates=[...document.querySelectorAll('[data-template].on')].map(x=>x.dataset.template);state=await api('/api/routing',{method:'POST',body:JSON.stringify({deployment_id:state.deployment_id,profile:state.profile,templates,expected_revision:state.state.routing_revision})});render();toast('Routing desired state saved; release consumers are pending.')}catch(e){toast(e.message)}};
function toggleAll(selector,on){if(!state)return toast('Load a deployment first.');document.querySelectorAll(selector).forEach(x=>{x.classList.toggle('on',on);x.querySelector('.check').textContent=on?'✓':''})}
document.querySelector('#select-all-specialists').onclick=()=>toggleAll('[data-pack],[data-specialist]',true);document.querySelector('#deselect-all-specialists').onclick=()=>toggleAll('[data-pack],[data-specialist]',false);document.querySelector('#select-all-routing').onclick=()=>toggleAll('[data-template][data-enabled="true"]',true);document.querySelector('#deselect-all-routing').onclick=()=>toggleAll('[data-template]',false);
function openPack(name=''){editingPack=name||null;let current=packDefs.find(x=>x.name===name);document.querySelector('#pack-name').value=current?current.name:'';document.querySelector('#pack-name').readOnly=Boolean(current);document.querySelector('#pack-description').value=current?current.description:'';document.querySelector('#pack-members').value=current?current.specialists.join(', '):'';document.querySelector('#pack-modal').classList.add('on')}
async function deletePack(name){let confirmation=prompt(`Deleting "${name}" only removes a local pack definition. Type the exact pack name to continue.`);if(confirmation===null)return;try{await api('/api/packs/delete',{method:'POST',body:JSON.stringify({name,confirmation})});await loadPacks();if(state)render();toast('Pack deleted.')}catch(e){toast(e.message)}}
document.querySelector('#new-pack').onclick=()=>openPack();document.querySelectorAll('[data-close]').forEach(x=>x.onclick=()=>document.querySelector('#pack-modal').classList.remove('on'));document.querySelector('#save-pack').onclick=async()=>{try{await api('/api/packs',{method:'POST',body:JSON.stringify({name:document.querySelector('#pack-name').value,description:document.querySelector('#pack-description').value,specialists:document.querySelector('#pack-members').value.split(',').map(x=>x.trim()).filter(Boolean)})});document.querySelector('#pack-modal').classList.remove('on');editingPack=null;await loadPacks();if(state)render();toast('Pack saved; it is not deployed until selected.')}catch(e){toast(e.message)}};document.querySelector('#copy').onclick=()=>navigator.clipboard.writeText(document.querySelector('#cli-command').textContent).then(()=>toast('CLI command copied.'));if(['overview','specialists','routing','packs','review'].includes(requestedView))switchTo(requestedView);Promise.all([loadPacks(),loadCatalog()]).catch(e=>toast(e.message));
</script></body></html>"""

_OBSERVABILITY_SECTION = r"""
<section id="observability" class="view"><p class="ey">Optional monitoring add-on</p><div class="head"><div><h1>Observability plan</h1><p class="sub">Choose a monitoring posture and a handoff destination. This browser saves only local desired state; it does not inspect an AWS account or create alarms.</p></div><button class="btn primary" id="save-observability">Save observability plan</button></div><div class="grid"><div class="card"><div class="cardhead"><h2>Monitoring posture</h2><p>Start with stock console links, then opt into a candidate baseline or enhanced plan.</p></div><div class="pad"><div class="field"><label for="observability-mode">Mode</label><select id="observability-mode"><option value="links">Links only — no new monitoring resources</option><option value="baseline">Baseline — dashboard and operational alarms</option><option value="enhanced">Enhanced — baseline plus business-health checks</option></select></div><div class="field"><label for="notification-kind">Notification handoff</label><select id="notification-kind"><option value="none">None — console links and alarms only</option><option value="sns">Amazon SNS topic ARN</option><option value="eventbridge">Amazon EventBridge event bus ARN</option></select></div><div class="field" id="notification-target-field"><label for="notification-target">Destination ARN</label><input id="notification-target" autocomplete="off" spellcheck="false" placeholder="arn:aws:sns:region:account:topic"></div></div><div class="detail"><h3>Quota gate stays in the CLI</h3><p>Before a future observability deploy, the CLI must check live dashboard and alarm counts, service quotas, and destination permissions. This local estimate is intentionally not an AWS preflight.</p></div></div><aside class="card"><div class="side"><p class="ey" style="color:#ddd6ff">Candidate footprint</p><h2 id="observability-title">Load a deployment</h2><p id="observability-note">A local estimate will appear here.</p><div class="stats"><div class="stat"><b id="observability-alarms">—</b><small>standard alarms</small></div><div class="stat"><b id="observability-dashboards">—</b><small>dashboards</small></div></div></div><div class="detail"><h3>Included coverage</h3><p id="observability-coverage">Load a deployment first.</p></div><div class="detail"><h3>Local planning only</h3><p id="observability-status">No AWS inventory, credentials, SDK calls, shell commands, or deployments are available here.</p></div></aside></div></section>
"""

_ONBOARDING_SECTIONS = r"""
<section id="welcome" class="view on"><div class="welcome"><p class="ey">Welcome to BADGERS</p><h1>Plan a deployment with confidence.</h1><p class="sub">This local planner prepares configuration for the deployment CLI. Open an existing local configuration or create one, choose capabilities and routing, then review the handoff. It never creates, changes, or deletes AWS resources.</p><div class="welcome-actions"><button class="btn primary" data-to="overview">Open or create configuration</button><button class="btn soft" data-to="tutorial">See the tutorial</button></div><div class="welcome-steps"><div class="welcome-step"><b>Choose a configuration</b><span>Open a saved deployment configuration or create one with a new ID and profile.</span></div><div class="welcome-step"><b>Set up capabilities</b><span>Select specialist packs, routing templates, and optional monitoring.</span></div><div class="welcome-step"><b>Review the handoff</b><span>Inspect the planned changes, then hand off to the CLI for preflight and deployment.</span></div></div></div></section>
<section id="tutorial" class="view"><p class="ey">How it works</p><div class="head"><div><h1>Configuration planner tutorial</h1><p class="sub">Use this screen to prepare a reviewed desired state. The CLI remains the only path that can make AWS changes.</p></div><button class="btn soft" data-to="welcome">Back to welcome</button></div><div class="tutorial-list"><div class="row"><span class="pill purple">1</span><div class="copy"><b>Start on Overview</b><small>Enter a lowercase deployment ID, choose Full, Headless, or Specialists, and click Load configuration.</small></div></div><div class="row"><span class="pill purple">2</span><div class="copy"><b>Choose capabilities</b><small>In Specialists, select packs or individual specialists. The planner flags container requirements automatically.</small></div></div><div class="row"><span class="pill purple">3</span><div class="copy"><b>Set workflow routing</b><small>Enable templates whose processors are selected. Invalid dependencies are shown before anything reaches the CLI.</small></div></div><div class="row"><span class="pill purple">4</span><div class="copy"><b>Optionally plan observability</b><small>Choose a local monitoring posture and notification handoff. AWS quota and permission checks happen later in CLI preflight.</small></div></div><div class="row"><span class="pill purple">5</span><div class="copy"><b>Review and deploy through the CLI</b><small>Review the required stages, copy the proposed command, and run it when you are ready.</small></div></div></div><div class="welcome-actions"><button class="btn primary" data-to="overview">Begin configuration</button></div></section>
"""

_OBSERVABILITY_SCRIPT = r"""
function renderObservabilityReview(plan){if(!state||!state.observability)return;let o=state.observability,r=o.candidate_plan.resources,host=document.querySelector('#plan-list');let review=document.querySelector('#observability-review');if(!review){review=document.createElement('div');review.id='observability-review';review.className='detail';host.parentElement.appendChild(review)}let notification=o.notification.kind==='none'?'No notification handoff':`${o.notification.kind.toUpperCase()} handoff selected`;review.innerHTML=`<h3>Observability add-on — ${esc(o.mode)}</h3><p>${r.cloudwatch_dashboards} dashboard, ${r.standard_alarms} standard alarms, ${r.composite_alarms} composite alarms. ${esc(notification)}. ${esc(plan.observability_handoff)}</p>`;}
function renderObservability(){if(!state||!state.observability)return;let o=state.observability,p=o.candidate_plan,r=p.resources;document.querySelector('#observability-mode').value=o.mode;document.querySelector('#notification-kind').value=o.notification.kind;document.querySelector('#notification-target').value=o.notification.target||'';let disabled=o.notification.kind==='none';document.querySelector('#notification-target-field').classList.toggle('hidden',disabled);document.querySelector('#notification-target').disabled=disabled;document.querySelector('#observability-title').textContent=`${o.mode[0].toUpperCase()+o.mode.slice(1)} monitoring`;document.querySelector('#observability-note').textContent=p.note;document.querySelector('#observability-alarms').textContent=r.standard_alarms;document.querySelector('#observability-dashboards').textContent=r.cloudwatch_dashboards;document.querySelector('#observability-coverage').textContent=p.coverage.join(' · ');document.querySelector('#observability-status').textContent=`Local desired-state revision ${o.revision}. ${o.in_sync?'Applied state matches.':'No observability deployment has been recorded.'}`;}
function updateNotificationTarget(){let disabled=document.querySelector('#notification-kind').value==='none';document.querySelector('#notification-target-field').classList.toggle('hidden',disabled);document.querySelector('#notification-target').disabled=disabled;if(disabled)document.querySelector('#notification-target').value='';}
document.querySelector('#notification-kind').onchange=updateNotificationTarget;
document.querySelector('#save-observability').onclick=async()=>{if(!state)return toast('Load a deployment first.');try{let mode=document.querySelector('#observability-mode').value,notification_kind=document.querySelector('#notification-kind').value,notification_target=document.querySelector('#notification-target').value;state=await api('/api/observability',{method:'POST',body:JSON.stringify({deployment_id:state.deployment_id,profile:state.profile,mode,notification_kind,notification_target,expected_revision:state.state.observability_revision})});render();toast('Observability plan saved locally. AWS quota checks remain a CLI preflight.')}catch(e){toast(e.message)}};
"""

_SPECIALIST_SELECTION_CSS = r"""
.selection-section + .selection-section .selection-head{margin-top:22px;border-top:1px solid var(--l)}.selection-head{display:flex;align-items:center;justify-content:space-between;gap:12px}.selection-toggle{display:inline-flex;align-items:center;gap:7px;color:#4d5566;font-size:12px;font-weight:750;cursor:pointer;white-space:nowrap}.row.select:hover{background:#f6fcf8}.row.on{background:#f3fbf6}.row:not(.on) .check,.selection-toggle input:not(:checked):not(:indeterminate){border-color:#b8bfcc;background-color:#fff;background-image:none}.selection-toggle input{appearance:none;width:16px;height:16px;margin:0;border:1px solid #b8bfcc;border-radius:4px;background:#fff;cursor:pointer}.selection-toggle input:checked{border-color:var(--cg-border);background-color:var(--cg);background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Cpath d='m3 8 3 3 7-7' fill='none' stroke='white' stroke-linecap='round' stroke-linejoin='round' stroke-width='2'/%3E%3C/svg%3E")}.selection-toggle input:indeterminate{border-color:var(--cg-border);background-color:var(--cg);background-image:linear-gradient(white,white);background-repeat:no-repeat;background-position:center;background-size:8px 2px}.btn.danger{border-color:#b14444;color:#8e3030}.btn.danger:hover{background:#fff1f1}.empty{padding:22px 18px;color:var(--m);font-size:12px}.head>div:last-child{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:7px}.welcome{max-width:780px;margin:36px auto}.welcome .sub{max-width:680px;font-size:15px}.welcome-actions{display:flex;flex-wrap:wrap;gap:8px;margin:24px 0}.welcome-steps{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.welcome-step{padding:16px;border:1px solid var(--l);border-radius:10px;background:#fff;box-shadow:0 7px 23px #252b3b0b}.welcome-step b,.welcome-step span{display:block}.welcome-step b{margin-bottom:5px}.welcome-step span{color:var(--m);font-size:12px}.tutorial-list{overflow:hidden;border:1px solid var(--l);border-radius:11px;background:#fff;box-shadow:0 7px 23px #252b3b0b}.tutorial-list .row:last-child{border-bottom:0}@media(max-width:760px){.head{flex-direction:column}.head>div:last-child{justify-content:flex-start}.welcome{margin:8px auto}.welcome-steps{grid-template-columns:1fr}}
"""

_INFORMATION_ARCHITECTURE_CSS = r"""
.bar-meta{display:flex;align-items:center;gap:12px}.bar-meta #active-config{font-size:12px;font-weight:700;color:#fff}.bar-meta small{color:#d3cdef}.nav-group{margin:20px 11px 5px;color:var(--m);font-size:10px;font-weight:800;letter-spacing:.09em;text-transform:uppercase}.saved-configurations{margin-top:22px;border-top:1px solid var(--l);padding-top:14px}.saved-configurations h3{margin:0 0 8px;font-size:12px}.saved-configuration{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:11px 0;border-top:1px solid #eff1f5}.saved-configuration:first-of-type{border-top:0}.row-actions{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:7px}@media(max-width:760px){.bar-meta #active-config{display:none}.saved-configuration{align-items:flex-start;flex-direction:column}.row-actions{justify-content:flex-start}.bar-meta{gap:0}}
"""

_SPECIALIST_SELECTION_SCRIPT = r"""
function setSelectionRow(row,on){row.classList.toggle('on',on);row.querySelector('.check').textContent=on?'✓':''}
function syncSectionToggle(selector,controlId){let rows=[...document.querySelectorAll(selector)],control=document.querySelector(controlId);if(!control)return;let selected=rows.filter(row=>row.classList.contains('on')).length;control.checked=rows.length>0&&selected===rows.length;control.indeterminate=selected>0&&selected<rows.length;control.disabled=rows.length===0}
function syncSpecialistSectionToggles(){syncSectionToggle('[data-pack]','#select-all-packs');syncSectionToggle('[data-specialist]','#select-all-direct-specialists')}
function toggleSelectionSection(selector,on){if(!state)return toast('Load a deployment first.');document.querySelectorAll(selector).forEach(row=>setSelectionRow(row,on));syncSpecialistSectionToggles()}
function renderSpecialistSelections(packRows,directRows){let host=document.querySelector('#specialist-list');host.innerHTML=`<div class="selection-section"><div class="cardhead selection-head"><h2>Packs</h2><label class="selection-toggle"><input id="select-all-packs" type="checkbox">Select all</label></div>${packRows}</div><div class="selection-section"><div class="cardhead selection-head"><h2>Individual specialists</h2><label class="selection-toggle"><input id="select-all-direct-specialists" type="checkbox">Select all</label></div>${directRows}</div>`;host.querySelectorAll('.row').forEach(row=>row.onclick=()=>{setSelectionRow(row,!row.classList.contains('on'));syncSpecialistSectionToggles()});document.querySelector('#select-all-packs').onchange=event=>toggleSelectionSection('[data-pack]',event.currentTarget.checked);document.querySelector('#select-all-direct-specialists').onchange=event=>toggleSelectionSection('[data-specialist]',event.currentTarget.checked);syncSpecialistSectionToggles()}
"""

_CONFIGURATION_INDEX_SCRIPT = r"""
const profileName=profile=>({full:'Full experience',headless:'Headless agent',specialists:'Specialists only'}[profile]||profile);
let savedConfigurations=[];
function resetLoadedConfiguration(){state=null;document.querySelector('#active-config').textContent='No configuration loaded';document.querySelector('#title-id').textContent='Not loaded';document.querySelector('#summary').textContent='Open or create a configuration to begin.';document.querySelector('#s-count').textContent='—';document.querySelector('#w-count').textContent='—';document.querySelector('#resolved').textContent='— specialists';document.querySelector('#container').textContent='Open a configuration first.';document.querySelector('#specialist-list').innerHTML='<div class="empty">Open a configuration from Deployment details to choose specialists.</div>';document.querySelector('#routing-list').innerHTML='<div class="empty">Open a configuration from Deployment details to choose workflow templates.</div>';document.querySelector('#routing-error').classList.add('hidden');document.querySelector('#plan-summary').textContent='Open a configuration to calculate the plan.';document.querySelector('#plan-list').innerHTML='';document.querySelector('#cli-command').textContent=`./deploy.sh ${document.querySelector('#profile').value} resume`;document.querySelector('#apply-note').textContent='The CLI will ask you to select the deployment and confirm after fresh preflight.';document.querySelector('#observability-title').textContent='Open a configuration';document.querySelector('#observability-note').textContent='A local estimate will appear here.';document.querySelector('#observability-alarms').textContent='—';document.querySelector('#observability-dashboards').textContent='—';document.querySelector('#observability-coverage').textContent='Open a configuration first.';document.querySelector('#observability-status').textContent='No AWS inventory, credentials, SDK calls, shell commands, or deployments are available here.';document.querySelector('#observability-review')?.remove()}
function renderConfigurationList(){let host=document.querySelector('#configuration-list');if(!savedConfigurations.length){host.innerHTML='<div class="empty">No saved local configurations yet. Enter a new ID above to create one.</div>';return}host.innerHTML=savedConfigurations.map(item=>`<div class="saved-configuration"><div class="copy"><b>${esc(item.deployment_id)}</b><small>${esc(profileName(item.profile))} · local configuration</small></div><div class="row-actions"><button class="btn" data-open-configuration="${esc(item.deployment_id)}">Open</button><button class="btn danger" data-remove-configuration="${esc(item.deployment_id)}">Remove</button></div></div>`).join('');host.querySelectorAll('[data-open-configuration]').forEach(button=>button.onclick=()=>openSavedConfiguration(button.dataset.openConfiguration));host.querySelectorAll('[data-remove-configuration]').forEach(button=>button.onclick=()=>removeSavedConfiguration(button.dataset.removeConfiguration))}
async function loadConfigurations(){try{savedConfigurations=await api('/api/configurations');renderConfigurationList()}catch(e){toast(e.message)}}
async function openSavedConfiguration(deploymentId){let item=savedConfigurations.find(config=>config.deployment_id===deploymentId);if(!item)return;document.querySelector('#deployment').value=item.deployment_id;document.querySelector('#profile').value=item.profile;await load();switchTo('overview')}
async function removeSavedConfiguration(deploymentId){let confirmation=prompt(`Removing "${deploymentId}" deletes only its local configuration. AWS resources are not deleted. Type the exact deployment ID to continue.`);if(confirmation===null)return;try{await api('/api/configuration/delete',{method:'POST',body:JSON.stringify({deployment_id:deploymentId,confirmation})});if(state?.deployment_id===deploymentId){document.querySelector('#deployment').value='';resetLoadedConfiguration()}await loadConfigurations();toast('Local configuration removed.')}catch(e){toast(e.message)}}
document.querySelector('#delete-configuration').onclick=()=>{let deploymentId=state?.deployment_id||document.querySelector('#deployment').value;if(!deploymentId)return toast('Enter a deployment ID first.');removeSavedConfiguration(deploymentId)};
loadConfigurations();
"""

_HTML = (
    _HTML.replace(
        '<button class="nav on" data-v="overview">Overview</button><button class="nav" data-v="specialists">Specialists</button><button class="nav" data-v="routing">Workflow routing</button><button class="nav" data-v="packs">Pack library</button><button class="nav" data-v="review">Review &amp; publish</button>',
        '<button class="nav" data-v="overview">1. Deployment details</button><button class="nav" data-v="specialists">2. Specialists</button><button class="nav" data-v="routing">3. Workflow routing</button><button class="nav" data-v="observability">4. Observability <span>(optional)</span></button><button class="nav" data-v="review">5. Review &amp; CLI handoff</button><p class="nav-group">Manage</p><button class="nav" data-v="packs">Pack library</button><button class="nav" data-v="tutorial">How it works</button>',
    )
    .replace(
        '<main>\n<section id="overview" class="view on">',
        '<main>\n' + _ONBOARDING_SECTIONS + '\n<section id="overview" class="view">',
    )
    .replace(
        '<small id="identity">loopback only · no AWS actions</small>',
        '<div class="bar-meta"><span id="active-config">No configuration loaded</span><small id="identity">local only · no AWS actions</small></div>',
    )
    .replace(
        '<h1>Configure before deploying</h1>',
        '<h1>Deployment details</h1>',
    )
    .replace(
        'A local planning surface for packs, specialists, and reviewed workflow templates. AWS changes remain in the CLI.',
        'Open a saved local configuration or create one with a new deployment ID. AWS changes remain in the CLI.',
    )
    .replace(
        '<h2>Choose a deployment</h2><p>New configuration creates only local state; it creates no AWS resources.</p>',
        '<h2>Open or create a configuration</h2><p>Opening a new deployment ID creates only local state; it creates no AWS resources.</p>',
    )
    .replace(
        '<label>Deployment ID</label><input id="deployment" value="beta">',
        '<label for="deployment">Deployment ID</label><input id="deployment" value="beta" maxlength="16" pattern="[a-z][a-z0-9-]{0,15}" autocomplete="off" spellcheck="false">',
    )
    .replace(
        '<label>Product profile</label><select id="profile">',
        '<label for="profile">Product profile</label><select id="profile">',
    )
    .replace(
        '<div id="specialist-list"></div>',
        '<div id="specialist-list" class="empty">Open a configuration from Deployment details to choose specialists.</div>',
    )
    .replace(
        '<div id="routing-list"></div>',
        '<div id="routing-list" class="empty">Open a configuration from Deployment details to choose workflow templates.</div>',
    )
    .replace(
        '<button class="btn primary" id="load">Load configuration</button>',
        '<button class="btn primary" id="load">Open configuration</button> <button class="btn danger" id="delete-configuration">Remove local configuration</button><div class="saved-configurations"><h3>Saved local configurations</h3><div id="configuration-list" class="empty">Loading saved local configurations…</div></div>',
    )
    .replace(
        '<div><button class="btn" id="select-all-specialists">Select all</button> <button class="btn" id="deselect-all-specialists">Deselect all</button> <button class="btn primary" id="save-selection">Save selection</button></div>',
        '<div><button class="btn primary" id="save-selection">Save selection</button></div>',
    )
    .replace(
        "document.querySelector('#specialist-list').innerHTML='<div class=\"cardhead\"><h2>Packs</h2></div>'+packRows+'<div class=\"cardhead\"><h2>Individual specialists</h2></div>'+directRows;\ndocument.querySelectorAll('#specialist-list .row').forEach(x=>x.onclick=()=>{x.classList.toggle('on');x.querySelector('.check').textContent=x.classList.contains('on')?'✓':''});",
        "renderSpecialistSelections(packRows,directRows);",
    )
    .replace(
        "document.querySelector('#select-all-specialists').onclick=()=>toggleAll('[data-pack],[data-specialist]',true);document.querySelector('#deselect-all-specialists').onclick=()=>toggleAll('[data-pack],[data-specialist]',false);document.querySelector('#select-all-routing').onclick=()=>toggleAll('[data-template][data-enabled=\"true\"]',true);document.querySelector('#deselect-all-routing').onclick=()=>toggleAll('[data-template]',false);",
        "document.querySelector('#select-all-routing').onclick=()=>toggleAll('[data-template][data-enabled=\"true\"]',true);document.querySelector('#deselect-all-routing').onclick=()=>toggleAll('[data-template]',false);",
    )
    .replace(
        "document.querySelector('#load').onclick=load;",
        "document.querySelector('#load').onclick=load;async function deleteConfiguration(){let deploymentId=state?.deployment_id||document.querySelector('#deployment').value;if(!deploymentId)return toast('Enter a deployment ID first.');let confirmation=prompt(`Deleting \"${deploymentId}\" removes only its local desired state. AWS resources are not deleted. Type the exact deployment ID to continue.`);if(confirmation===null)return;try{await api('/api/configuration/delete',{method:'POST',body:JSON.stringify({deployment_id:deploymentId,confirmation})});state=null;document.querySelector('#deployment').value='';document.querySelector('#title-id').textContent='Not loaded';document.querySelector('#summary').textContent='Choose a deployment to begin.';document.querySelector('#s-count').textContent='—';document.querySelector('#w-count').textContent='—';document.querySelector('#resolved').textContent='— specialists';document.querySelector('#container').textContent='Select a deployment first.';document.querySelector('#specialist-list').innerHTML='';document.querySelector('#routing-list').innerHTML='';document.querySelector('#routing-error').classList.add('hidden');document.querySelector('#plan-summary').textContent='Load a deployment to calculate the plan.';document.querySelector('#plan-list').innerHTML='';toast('Local configuration deleted.')}catch(e){toast(e.message)}}document.querySelector('#delete-configuration').onclick=deleteConfiguration;",
    )
    .replace(
        "document.querySelector('#specialist-list').innerHTML='';document.querySelector('#routing-list').innerHTML='';",
        "document.querySelector('#specialist-list').innerHTML='<div class=\"empty\">Open a configuration from Deployment details to choose specialists.</div>';document.querySelector('#routing-list').innerHTML='<div class=\"empty\">Open a configuration from Deployment details to choose workflow templates.</div>';",
    )
    .replace(
        "document.querySelector('#plan-list').innerHTML='';toast('Local configuration deleted.')",
        "document.querySelector('#plan-list').innerHTML='';document.querySelector('#cli-command').textContent=`./deploy.sh ${document.querySelector('#profile').value} resume`;document.querySelector('#apply-note').textContent='The CLI will ask you to select the deployment and confirm after fresh preflight.';document.querySelector('#observability-title').textContent='Load a deployment';document.querySelector('#observability-note').textContent='A local estimate will appear here.';document.querySelector('#observability-alarms').textContent='—';document.querySelector('#observability-dashboards').textContent='—';document.querySelector('#observability-coverage').textContent='Load a deployment first.';document.querySelector('#observability-status').textContent='No AWS inventory, credentials, SDK calls, shell commands, or deployments are available here.';toast('Local configuration deleted.')",
    )
    .replace(
        '<button class="nav" data-v="review">Review &amp; publish</button>',
        '<button class="nav" data-v="observability">Observability</button>'
        '<button class="nav" data-v="review">Review &amp; publish</button>',
    )
    .replace("</main></div><div id=\"pack-modal\"", _OBSERVABILITY_SECTION + "</main></div><div id=\"pack-modal\"")
    .replace("renderPlan()}", "renderPlan();renderObservability()}")
    .replace(
        "document.querySelector('#apply-note').textContent=p.apply_note",
        "document.querySelector('#apply-note').textContent=p.apply_note;renderObservabilityReview(p)",
    )
    .replace(
        "function render(){if(!state)return;let s=state.selection,route=state.routing;",
        "function render(){if(!state)return;document.querySelector('#active-config').textContent=`Editing: ${state.deployment_id} · ${profileName(state.profile)}`;let s=state.selection,route=state.routing;",
    )
    .replace(
        "['overview','specialists','routing','packs','review']",
        "['welcome','tutorial','overview','specialists','routing','packs','observability','review']",
    )
    .replace("</style>", _SPECIALIST_SELECTION_CSS + _INFORMATION_ARCHITECTURE_CSS + "\n</style>")
    .replace("</script></body></html>", _SPECIALIST_SELECTION_SCRIPT + "\n</script></body></html>")
    .replace("</script></body></html>", _OBSERVABILITY_SCRIPT + "\n</script></body></html>")
    .replace("</script></body></html>", _CONFIGURATION_INDEX_SCRIPT + "\n</script></body></html>")
)


class ConfigurationServer(ThreadingHTTPServer):
    """A private loopback server carrying its per-process browser capability."""

    allow_reuse_address = False

    def __init__(self, repo_root: Path) -> None:
        super().__init__(("127.0.0.1", 0), ConfigurationHandler)
        self.repo_root = repo_root
        self.session_token = secrets.token_urlsafe(32)
        self.origin = f"http://127.0.0.1:{self.server_port}"


class ConfigurationHandler(BaseHTTPRequestHandler):
    server: ConfigurationServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        """Do not log local request URLs, including the session capability."""

    def _headers(self, status: HTTPStatus, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; connect-src 'self'; img-src 'self'; "
            "style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'",
        )

    def _json(self, status: HTTPStatus, value: object) -> None:
        data = json.dumps(value).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _require_local_session(self, *, require_origin: bool) -> bool:
        if self.headers.get("X-Badgers-Session") != self.server.session_token:
            self._json(HTTPStatus.FORBIDDEN, {"error": "Local session verification failed."})
            return False
        if require_origin and self.headers.get("Origin") != self.server.origin:
            self._json(HTTPStatus.FORBIDDEN, {"error": "Unexpected request origin."})
            return False
        return True

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > _MAX_BODY_BYTES:
            raise UIError("Request body is missing or too large.")
        try:
            value = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            raise UIError("Request body must be JSON.") from error
        if not isinstance(value, dict):
            raise UIError("Request body must be a JSON object.")
        return value

    def _route(self, action: str, payload: dict[str, Any] | None = None) -> object:
        root = self.server.repo_root
        query = parse_qs(urlparse(self.path).query)
        requested_profile = str(query.get("profile", [""])[0]) or None
        if action == "packs":
            return packs(root, requested_profile)
        if action == "catalog":
            return catalog(root)
        if action == "configurations":
            return configurations(root)
        if action == "save_pack":
            return save_pack(root, payload or {})
        if action == "delete_pack":
            return delete_pack(root, payload or {})
        if action == "selection":
            return save_selection(root, payload or {})
        if action == "routing":
            return save_routing(root, payload or {})
        if action == "observability":
            return save_observability(root, payload or {})
        if action == "delete_configuration":
            return delete_configuration(root, payload or {})
        deployment_id = str(query.get("id", [""])[0])
        profile = str(query.get("profile", [""])[0])
        if action == "configuration":
            return configuration(root, deployment_id, profile)
        if action == "plan":
            return plan(root, deployment_id, profile)
        raise UIError("Unknown local configuration endpoint.")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            token = parse_qs(parsed.query).get("token", [""])[0]
            if not secrets.compare_digest(token, self.server.session_token):
                self._json(HTTPStatus.FORBIDDEN, {"error": "Local session verification failed."})
                return
            data = _HTML.encode("utf-8")
            self._headers(HTTPStatus.OK, "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        # Browser GET requests may omit Origin or vary it during local navigation.
        # Reads remain capability-protected; all mutations require exact origin.
        if not self._require_local_session(require_origin=False):
            return
        try:
            if parsed.path == "/api/packs":
                self._json(HTTPStatus.OK, self._route("packs"))
            elif parsed.path == "/api/catalog":
                self._json(HTTPStatus.OK, self._route("catalog"))
            elif parsed.path == "/api/configurations":
                self._json(HTTPStatus.OK, self._route("configurations"))
            elif parsed.path == "/api/configuration":
                self._json(HTTPStatus.OK, self._route("configuration"))
            elif parsed.path == "/api/plan":
                self._json(HTTPStatus.OK, self._route("plan"))
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
        except (
            UIError,
            observability_configuration.ObservabilityConfigurationError,
            specialist_selection.SelectionError,
            routing_configuration.RoutingConfigurationError,
        ) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def do_POST(self) -> None:  # noqa: N802
        # State-changing requests require the capability and an exact origin.
        if not self._require_local_session(require_origin=True):
            return
        try:
            payload = self._body()
            routes = {
                "/api/selection": "selection",
                "/api/routing": "routing",
                "/api/observability": "observability",
                "/api/packs": "save_pack",
                "/api/packs/delete": "delete_pack",
                "/api/configuration/delete": "delete_configuration",
            }
            action = routes.get(urlparse(self.path).path)
            if not action:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
                return
            self._json(HTTPStatus.OK, self._route(action, payload))
        except (
            UIError,
            observability_configuration.ObservabilityConfigurationError,
            specialist_selection.SelectionError,
            routing_configuration.RoutingConfigurationError,
        ) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo.resolve()
    if not (repo_root / "deploy.sh").is_file():
        parser.error(f"Not a BADGERS repository root: {repo_root}")
    server = ConfigurationServer(repo_root)
    url = f"{server.origin}/?token={server.session_token}"
    print("BADGERS local configuration UI")
    print("Listening only on 127.0.0.1; no AWS actions are available from the UI.")
    print(f"Open: {url}")
    print("Press Ctrl-C to stop.")
    if not args.no_browser:
        threading.Timer(0.15, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped local configuration UI.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
