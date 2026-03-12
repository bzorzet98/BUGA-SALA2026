import json
from pathlib import Path
from typing import Union

def load_config(config_path: Union[str, Path]) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    required_top_keys = ["paths", "dataset", "model", "train", "val"]
    for key in required_top_keys:
        if key not in config:
            raise ValueError(f"Falta la sección '{key}' en el config")

    return config