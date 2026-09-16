"""How account and Region appear inside a cdk-nag ``appliesTo`` string.

cdk-nag reports findings against the **resolved** CloudFormation template, so a policy ARN
built with ``self.account`` does not always render the same way:

* When the stack has a concrete environment -- which it does on the deploy path, because
  ``scripts/common.sh`` exports ``CDK_DEFAULT_ACCOUNT`` and ``CDK_DEFAULT_REGION`` -- the
  template holds the literal 12-digit account and the Region name, and the finding reads
  ``arn:aws:logs:us-west-2:123456789012:...``.
* When the stack is environment-agnostic, the template holds ``{"Ref": "AWS::AccountId"}``
  and cdk-nag prints ``arn:aws:logs:<AWS::Region>:<AWS::AccountId>:...``.

An ``appliesTo`` entry is an exact string match, so it matches one form or the other and
never both. Suppressions that hardcoded ``<AWS::AccountId>`` therefore matched nothing on the
deploy path, which is how several of them sat dead in the tree. Emitting both forms costs a
few list entries and removes the failure mode.
"""

from __future__ import annotations

import json
from typing import Any

from aws_cdk import Stack, Token
from constructs import IConstruct

__all__ = ["account_renderings", "nag_resource_string", "region_renderings"]


def nag_resource_string(scope: IConstruct, value: Any) -> str:
    """Render ``value`` exactly as cdk-nag will print it in an ``AwsSolutions-IAM5``
    finding, so it can be used verbatim in ``appliesTo``.

    cdk-nag resolves the value against the stack and flattens CloudFormation intrinsics
    with a fixed algorithm (``lib/utils/flatten-cfn-reference.ts``):

    * ``Fn::Join``        -> items flattened and joined with the delimiter
    * ``Fn::Sub``         -> the template string, ``${x}`` rewritten to ``<x>``
    * ``Fn::GetAtt``      -> ``<LogicalId.Attribute>``
    * ``Fn::ImportValue`` -> the export name, flattened (the ``Fn::ImportValue`` wrapper
      itself contributes nothing)
    * ``Ref``             -> ``<LogicalId>``, which is how ``<AWS::AccountId>`` arises

    Reproducing that here means an ``appliesTo`` entry can be built from the same token the
    policy statement was built from -- an imported bucket name, a cross-stack export, a
    ``GetAtt`` -- and is guaranteed to match, instead of being transcribed from a finding
    and going stale the next time a logical ID or export name changes.
    """
    resolved = Stack.of(scope).resolve(value)
    return _flatten(resolved)


def _flatten(node: Any) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return node.replace("${", "<").replace("}", ">")
    if isinstance(node, dict):
        if "Fn::Join" in node:
            delimiter, items = node["Fn::Join"]
            return delimiter.join(_flatten(i) for i in items)
        if "Fn::Sub" in node:
            return _flatten(node["Fn::Sub"])
        if "Fn::GetAtt" in node:
            resource, attribute = node["Fn::GetAtt"]
            return f"<{_flatten(resource)}.{_flatten(attribute)}>"
        if "Fn::ImportValue" in node:
            return _flatten(node["Fn::ImportValue"])
        if "Ref" in node:
            return f"<{_flatten(node['Ref'])}>"
    return json.dumps(node)


def _both(resolved: str, pseudo: str) -> list[str]:
    """``resolved`` first, then the pseudo-parameter form, with duplicates dropped.

    When the stack is environment-agnostic, ``Stack.account`` / ``Stack.region`` are
    *tokens*, not strings. Interpolating a token into an ``appliesTo`` entry would make CDK
    resolve that metadata entry to an ``Fn::Join`` object at synth -- a non-string cdk-nag
    cannot match and may reject. In that case only the pseudo form is meaningful, so the
    token is dropped rather than emitted.
    """
    if Token.is_unresolved(resolved):
        return [pseudo]
    return list(dict.fromkeys([resolved, pseudo]))


def account_renderings(account: str) -> list[str]:
    """Both ways ``Stack.account`` can appear in a cdk-nag finding."""
    return _both(account, "<AWS::AccountId>")


def region_renderings(region: str) -> list[str]:
    """Both ways ``Stack.region`` can appear in a cdk-nag finding."""
    return _both(region, "<AWS::Region>")
