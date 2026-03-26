from __future__ import annotations

from pathlib import Path

import yaml


def load_defaults(entry_point: str = "main") -> dict[str, str]:
    """Load default parameter values from MLproject.

    Reads the MLproject YAML file and extracts the default values for the
    given entry point. Returns a dict suitable for use as click's
    default_map. All values are stringified so click can parse them
    through its type system.

    :param entry_point: Entry point name (e.g. 'main', 'test')
    :return: Dict mapping option names (with hyphens) to string defaults
    """
    mlproject_path = Path(__file__).parent.parent.parent / "MLproject"
    if not mlproject_path.exists():
        return {}
    data = yaml.safe_load(mlproject_path.read_text())
    params = data["entry_points"][entry_point]["parameters"]
    defaults = {}
    for k, v in params.items():
        val = v["default"]
        defaults[k] = str(val) if val != "" else None
    return defaults
