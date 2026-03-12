import json
from pathlib import Path


def load_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = ["video_path", "model_path", "output_dir", "class_names"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Falta la llave requerida en config.json: {key}")

    return config