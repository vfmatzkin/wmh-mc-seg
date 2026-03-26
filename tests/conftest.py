import pytest
import yaml


@pytest.fixture
def mlproject_file(tmp_path):
    """Minimal MLproject YAML with 'main' and 'test' entry points."""
    data = {
        'name': 'TestProject',
        'entry_points': {
            'main': {
                'parameters': {
                    'epochs': {'type': 'int', 'default': 100},
                    'batch_size': {'type': 'int', 'default': 8},
                    'lr': {'type': 'float', 'default': 0.001},
                    'loss': {'type': 'string', 'default': 'dice'},
                }
            },
            'test': {
                'parameters': {
                    'batch_size': {'type': 'int', 'default': 1},
                    'patch_size': {'type': 'int', 'default': 32},
                    'seed': {'type': 'int', 'default': 42},
                }
            },
        },
    }
    path = tmp_path / 'MLproject'
    path.write_text(yaml.dump(data))
    return path
