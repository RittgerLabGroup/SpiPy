#!/usr/bin/env python3
"""Preview CURC workflow submission payloads from a user config file."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.curc.runner import plan_submissions


def load_config(config_path: Path):
    spec = spec_from_file_location("curc_user_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config module from {config_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CONFIG"):
        raise ValueError(f"Config module {config_path} must define CONFIG")
    return module.CONFIG


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: submit_curc_sensor_workflow.py <config.py>", file=sys.stderr)
        return 2

    config_path = Path(argv[1]).expanduser().resolve()
    config = load_config(config_path)
    payloads = plan_submissions(config)
    rendered = []
    for payload in payloads:
        rendered.append(asdict(payload) if is_dataclass(payload) else payload)
    print(json.dumps(rendered, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
