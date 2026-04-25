import yaml

def load_config(path: str = "/srv/THESIS/energy_profiling_thesis/configs/config.yaml") -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config