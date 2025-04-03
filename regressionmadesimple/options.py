from types import SimpleNamespace
import json
import os

class RecursiveNamespace(SimpleNamespace):
    """
    Solutiuon found on dev.to: https://dev.to/taqkarim/extending-simplenamespace-for-nested-dictionaries-58e8
    Recursively set SimpleNamespace attributes for nested dictionaries.
    """
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)
        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if isinstance(val, dict):
                setattr(self, key, RecursiveNamespace(**val))
            elif isinstance(val, list):
                setattr(self, key, list(map(self.map_entry, val)))

# Default config
_raw_options = {
    "plot": {
        "backend": "plotly"
    },
    "linear": {
        "auto_split": True,
        "show_summary": True
    },
    "training": {
        "random_state": 1,
        "test_size": 0.15
    }
}

options = RecursiveNamespace(**_raw_options)

def namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, list):
        return [namespace_to_dict(v) for v in ns]
    else:
        return ns

def save_options(path="rms_config.json"):
    with open(path, "w") as f:
        json.dump(namespace_to_dict(options), f, indent=2)

def load_options(path="rms_config.json"):
    global options
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            options = RecursiveNamespace(**data)

def reset_options():
    global options
    options = RecursiveNamespace(**_raw_options)
