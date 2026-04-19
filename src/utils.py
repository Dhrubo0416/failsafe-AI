"""
utils.py — FailSafe AI Utility Functions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Helper functions for config loading, serialization, and data validation.
"""

import yaml
import joblib
import os

# Resolve config path relative to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "config.yaml")


def load_config() -> dict:
    """Load the YAML configuration file."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT


def resolve_path(relative_path: str) -> str:
    """Resolve a relative path against the project root."""
    return os.path.join(_PROJECT_ROOT, relative_path)


def pickle_dump(data, relative_path: str) -> None:
    """Serialize and save a Python object using joblib."""
    full_path = resolve_path(relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    joblib.dump(data, full_path)


def pickle_load(relative_path: str):
    """Load a serialized Python object using joblib."""
    full_path = resolve_path(relative_path)
    return joblib.load(full_path)
