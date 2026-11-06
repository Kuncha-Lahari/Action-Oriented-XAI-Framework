"""
config.py
Central configuration loader for the Action-Oriented XAI Framework.

Reads ``config.yaml`` from the project root and exposes every section as
a typed attribute on a singleton ``CONFIG`` object. Any value in the YAML
can be overridden by an environment variable of the form:

    XAI_<SECTION>_<KEY>

e.g. ``XAI_SCM_CI_TEST=chisq`` overrides ``scm.ci_test``.

Design notes
------------
- The singleton is created at import time and cached. Modules should
  import ``CONFIG`` once and read attributes directly:
      from config import CONFIG
      alpha_schedule = CONFIG.scm.alpha_schedule
- Nested YAML sections are exposed as SimpleNamespace objects, so both
  attribute access (CONFIG.scm.ci_test) and dict-style access
  (CONFIG.scm["ci_test"]) work where reasonable.
- Environment overrides are applied shallowly: only top-level keys
  within a section are overridable. Deep nesting is not supported by
  design -- if we need it later, we can extend this.
- The config file is searched in this order:
      1. Path given by env var XAI_CONFIG_PATH
      2. ./config.yaml (project root, i.e. next to this file)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml

logger = logging.getLogger("config")

CONFIG_FILENAME = "config.yaml"
ENV_OVERRIDE_PREFIX = "XAI_"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _config_path() -> Path:
    """Resolve the config file path."""
    override = os.environ.get(f"{ENV_OVERRIDE_PREFIX}CONFIG_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / CONFIG_FILENAME


def _apply_env_overrides(section_name: str, section: Dict[str, Any]) -> Dict[str, Any]:
    """Override top-level keys in a section using environment variables.

    For section ``scm`` and key ``ci_test``, the env var is
    ``XAI_SCM_CI_TEST``. Values are parsed best-effort: YAML/JSON scalars
    (ints, floats, bools, null) are converted; anything else stays a string.
    """
    overridden = dict(section)
    prefix = f"{ENV_OVERRIDE_PREFIX}{section_name.upper()}_"
    for key in list(section.keys()):
        env_key = f"{prefix}{key.upper()}"
        if env_key not in os.environ:
            continue
        raw_value = os.environ[env_key]
        try:
            parsed = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            parsed = raw_value
        overridden[key] = parsed
        logger.info(
            "Config override: %s.%s = %r (from %s)",
            section_name,
            key,
            parsed,
            env_key,
        )
    return overridden


def _to_namespace(obj: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace for attribute access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _load_config() -> SimpleNamespace:
    """Load config.yaml, apply env overrides, and return a namespace."""
    path = _config_path()

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {path}. "
            "Create a config.yaml in the project root, or set "
            f"{ENV_OVERRIDE_PREFIX}CONFIG_PATH to point at one."
        )

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file at {path} must contain a top-level mapping; "
            f"got {type(raw).__name__}."
        )

    # Apply env overrides section by section.
    overridden: Dict[str, Any] = {}
    for section_name, section in raw.items():
        if isinstance(section, dict):
            overridden[section_name] = _apply_env_overrides(section_name, section)
        else:
            overridden[section_name] = section

    namespace = _to_namespace(overridden)
    logger.info("Configuration loaded from %s", path)
    return namespace


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

CONFIG: SimpleNamespace = _load_config()


# ---------------------------------------------------------------------------
# Convenience accessors (avoid attribute-typo crashes in downstream modules)
# ---------------------------------------------------------------------------

def get(section: str, key: str, default: Any = None) -> Any:
    """Safely fetch a config value with a default.

    Prefer direct attribute access in new code (CONFIG.scm.ci_test).
    This helper exists for callers that want a defensive lookup.
    """
    sec = getattr(CONFIG, section, None)
    if sec is None:
        return default
    return getattr(sec, key, default)


if __name__ == "__main__":
    # Quick manual check.
    import json

    def _dump(ns: Any) -> Any:
        if isinstance(ns, SimpleNamespace):
            return {k: _dump(v) for k, v in vars(ns).items()}
        if isinstance(ns, list):
            return [_dump(v) for v in ns]
        return ns

    print(json.dumps(_dump(CONFIG), indent=2, default=str))