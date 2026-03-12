from pathlib import Path
import yaml
from typing import List

def create_data_yaml(dataset_root: Path, class_names: List[str]):
    data_yaml = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)}
    }

    yaml_path = dataset_root / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    return yaml_path