"""Configuration schema and loader."""

from pathlib import Path

from fosim.config.schema import SimConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
STRESS_TEMPLATES_PATH = PROJECT_ROOT / "config" / "stress_templates.yaml"

__all__ = ["DEFAULT_CONFIG_PATH", "PROJECT_ROOT", "STRESS_TEMPLATES_PATH", "SimConfig", "load_config"]
