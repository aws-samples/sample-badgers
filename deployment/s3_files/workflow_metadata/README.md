[← BADGERS home](../../../README.md)

# Specialist capability metadata

`specialist_capabilities.json` maps stable capability tags to the product's
specialist IDs. The release stager filters this catalog to the specialists
selected by the profile. The deterministic planner resolves capabilities using
that filtered mapping, so a workflow never needs to embed a
deployment-specific Lambda name.

Capability metadata is part of the private Step 3 configuration release. It
does not independently make a specialist available to Headless or Specialists:
the selected profile and pack remain the authority. The initial Headless packs
therefore do not expose the Full product's native DOCX/PPTX processors. See the
[Release Guide](../../../RELEASE_GUIDE.md).

Adding a specialist to a workflow requires adding its capability metadata and
including the specialist in an eligible pack. Validation fails before release
creation if any referenced capability is unavailable.
