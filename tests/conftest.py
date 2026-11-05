"""Shared pytest configuration."""

import os

# litellm fetches its model capability map from the network when it is imported.
# Tests that read that map, such as the reasoning-effort model config tests, then
# fail when the upstream data changes. Use the map that ships with the package so the
# results depend only on the pinned litellm version. This must run before litellm is
# imported, so pytest must load this file first.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
