from pathlib import Path
from typing import Union

import yaml


def read_yaml(path: Union[Path, str]) -> dict:
    path = str(path)

    with open(path, 'r') as f:
        return yaml.safe_load(f)


def write_yaml(path: Union[Path, str], data: dict):
    path = str(path)

    with open(path, 'w') as f:
        yaml.dump(data, f)


def archive_settings(config):
    archive_settings_dir = Path(__file__).parent / 'config' / 'archive'
    archive_settings_dir.mkdir(parents=True, exist_ok=True)

    method_name = config['method_name']

    archive_settings_path = archive_settings_dir / (method_name + '.yaml')

    write_yaml(archive_settings_path, config)
