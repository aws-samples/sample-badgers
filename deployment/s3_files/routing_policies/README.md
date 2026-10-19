# BADGERS routing policies

Routing policies are immutable JSON release artifacts. They interpret
normalized classifier facts and add capability tags; they never reference
specialist names, Lambda ARNs, or customer executable code.

Each policy file name must equal `policy_id`. Its required capabilities must
exactly match the referenced workflow's invariant capabilities. Routes can add
only capabilities already approved by that workflow and available in the
selected specialist pack. Low-confidence behavior is explicit: it either adds
an approved fallback capability or fails planning before execution.
