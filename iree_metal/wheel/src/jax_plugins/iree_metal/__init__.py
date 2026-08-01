"""JAX plugin entry point for the bundled IREE-Metal preview runtime."""

from __future__ import annotations

import logging

import jax._src.xla_bridge as xb

from iree_metal_preview.runtime import configure_environment


logger = logging.getLogger(__name__)


def initialize():
    try:
        paths = configure_environment()
    except Exception as exc:
        logger.warning("iree-metal-preview registration skipped: %s", exc)
        return
    xb.register_plugin(
        "iree_metal",
        priority=500,
        library_path=str(paths["plugin"]),
        options={"COMPILER_LIB_PATH": str(paths["compiler"])},
    )
