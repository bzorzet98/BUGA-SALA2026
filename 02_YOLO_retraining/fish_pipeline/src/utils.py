from pathlib import Path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_stem(path_str: str) -> str:
    return Path(path_str).stem