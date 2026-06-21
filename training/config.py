"""
Tiny YAML config loader for training_orion_reg.py (and compare_models.py).

Usage in a script (module level, import-safe — does NOT consume another script's argv):
    from config import load_config
    CFG = load_config()                      # reads --config / --set from sys.argv
    lr = CFG.train.lr                        # dot access
    d  = CFG.to_dict()                       # plain nested dict (for merge / save)

CLI:
    python training_orion_reg.py                              # uses config.yaml
    python training_orion_reg.py --config configs/lora.yaml
    python training_orion_reg.py --set loss.lambda_bg=12 --set finetune.mode=unfreeze

--set values are parsed as YAML, so types come out right:
    loss.lambda_bg=12        -> int 12
    train.lr=3e-4            -> float
    train.oversample=true    -> bool
    data.val_slides=[CRC19,CRC30]  -> list

Programmatic (compare_models.py):
    from config import load_dict, deep_merge, save_config
    base = load_dict("config.yaml")
    arm  = deep_merge(base, {"finetune": {"mode": "lora"}})
    save_config(arm, "configs/_arm_lora.yaml")
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

DEFAULT_CONFIG = "config.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base` (override wins)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_path(d: dict, dotted: str, value) -> None:
    """In-place set d['a']['b']['c'] = value for dotted='a.b.c'."""
    keys = dotted.split(".")
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ValueError(f"--set {dotted}: '{k}' is not a mapping")
    node[keys[-1]] = value


def _coerce(s: str):
    """Parse a --set value as YAML so types are natural (int/float/bool/list/str)."""
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def load_dict(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def to_namespace(d):
    """Recursively convert a nested dict to a dot-accessible namespace (lists kept as-is)."""
    if isinstance(d, dict):
        ns = SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
        ns._dict = d  # keep the backing dict for to_dict()
        return ns
    return d


def _ns_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        # skip backing/meta attrs (_dict, _config_path) and the attached to_dict() lambda —
        # only real config keys (non-underscore, non-callable) go into the dumped dict.
        return {k: _ns_to_dict(v) for k, v in vars(ns).items()
                if not k.startswith("_") and not callable(v)}
    return ns


def save_config(cfg, path: str | Path) -> None:
    """Write a config (dict or namespace) to YAML."""
    d = cfg if isinstance(cfg, dict) else _ns_to_dict(cfg)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)


def load_config(default_path: str = DEFAULT_CONFIG, argv=None):
    """Load config.yaml (or --config) with --set overrides applied. Returns a dot-access
    namespace. Uses parse_known_args so importing this from another CLI script is safe."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", default=default_path)
    ap.add_argument("--set", action="append", default=[], dest="overrides",
                    metavar="a.b.c=value")
    args, _ = ap.parse_known_args(sys.argv[1:] if argv is None else argv)

    d = load_dict(args.config)
    for item in args.overrides:
        if "=" not in item:
            raise ValueError(f"--set expects a.b.c=value, got {item!r}")
        key, val = item.split("=", 1)
        set_path(d, key.strip(), _coerce(val.strip()))

    ns = to_namespace(d)
    ns._config_path = args.config
    ns.to_dict = lambda: _ns_to_dict(ns)  # type: ignore[attr-defined]
    return ns


# convenience: attach to_dict for nested namespaces too
def ns_to_dict(ns) -> dict:
    return _ns_to_dict(ns)