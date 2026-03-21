import yaml

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config