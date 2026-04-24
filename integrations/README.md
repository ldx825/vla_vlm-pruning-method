# Integrations

This directory is reserved for model-method integration patches and notes.

Suggested structure:

```text
integrations/
  openvla_oft/
    fastv/
    sparsevlm/
    fastervlm/
    visionzip/
    pyramiddrop/
    auto_prune/
  nora/
  mantis/
```

Keep upstream method code in `methods/` and model code in `models/`. Put glue code, patches, run scripts, and model-specific modifications here when the integration becomes stable enough to track.
