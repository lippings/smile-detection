from pathlib import Path
from typing import Union

import yaml

def read_yaml(path: Union[Path, str]) -> dict:
    path = str(path)

    with open(path, 'r') as f:
        return yaml.safe_load(f)
