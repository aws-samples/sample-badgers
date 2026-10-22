[← BADGERS home](../../../README.md)

# BADGERS routing policies

Routing policies are immutable JSON release artifacts. They interpret
normalized classifier facts and add capability tags; they never reference
specialist names, Lambda ARNs, or customer executable code.

The current Headless policy uses the classifier as the primary auto-routing
signal. It is deployed only as part of a SHA-pinned configuration release and
cannot be supplied or overridden by a request caller. See the
[Release Guide](../../../RELEASE_GUIDE.md) for the distinction between this
configuration artifact and a Git release tag.

Each policy file name must equal `policy_id`. Its required capabilities must
exactly match the referenced workflow's invariant capabilities. Routes can add
only capabilities already approved by that workflow and available in the
selected specialist pack. Low-confidence behavior is explicit: it either adds
an approved fallback capability or fails planning before execution.
