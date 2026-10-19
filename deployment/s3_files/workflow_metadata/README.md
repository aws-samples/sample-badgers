# Specialist capability metadata

`specialist_capabilities.json` maps stable capability tags to the product's
specialist IDs. The release stager filters this catalog to the specialists
selected by the profile. The deterministic planner resolves capabilities using
that filtered mapping, so a workflow never needs to embed a
deployment-specific Lambda name.

Adding a specialist to a workflow requires adding its capability metadata and
including the specialist in an eligible pack. Validation fails before release
creation if any referenced capability is unavailable.
