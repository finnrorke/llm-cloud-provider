#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def load_json(path_value, *, required=False):
    if not path_value:
        return {}

    path = Path(path_value)
    if not path.exists():
        if required:
            raise RuntimeError(f"Config file not found: {path}")
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Config JSON must be an object: {path}")
    return payload


def run_config_args(run_cfg):
    if not run_cfg:
        return {}
    args_obj = run_cfg.get("args", run_cfg)
    if not isinstance(args_obj, dict):
        raise RuntimeError("run-config JSON must be an object or contain an object under 'args'.")
    return args_obj


def resolve_path(path_value, *, base_dir):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()


def require_run_config_path(argv, *, script_name):
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise RuntimeError(f"Usage: {script_name} <run-config.json>")
    return args[0]
