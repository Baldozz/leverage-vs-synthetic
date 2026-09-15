"""Re-export of the configuration schema (implementation lives in ``fosim.config.schema``)."""

from fosim.config.schema import *  # noqa: F403
from fosim.config.schema import SimConfig, load_config

__all__ = ["SimConfig", "load_config"]
