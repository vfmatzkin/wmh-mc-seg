from pathlib import Path

import yaml


def load_defaults(entry_point='main'):
    """ Load default parameter values from MLproject

    Reads the MLproject YAML file and extracts the default values for the
    given entry point. Returns a dict suitable for use as click's
    default_map.

    :param entry_point: Entry point name (e.g. 'main', 'test')
    :return: Dict mapping option names to their default values
    """
    mlproject_path = Path(__file__).parent.parent.parent / 'MLproject'
    if not mlproject_path.exists():
        return {}
    data = yaml.safe_load(mlproject_path.read_text())
    params = data['entry_points'][entry_point]['parameters']
    return {k.replace('_', '-'): v['default'] for k, v in params.items()}
