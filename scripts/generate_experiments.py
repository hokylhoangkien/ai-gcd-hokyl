from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible fusion-mode/seed configs")
    parser.add_argument("--base", default="configs/planogram_proposed.json")
    parser.add_argument("--output", default="configs/generated")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    experiments = {
        "simgcd_style": {
            "fusion_mode": "visual",
            "lambda_dual_branch": 0.0, "lambda_cico": 0.0,
            "lambda_sdl": 0.0, "lambda_adl": 0.0, "lambda_hierarchy": 0.0,
            "lambda_calibration": 0.0, "lambda_attribute": 0.0, "lambda_conflict": 0.0,
        },
        "selex_adapted": {
            "fusion_mode": "visual",
            "lambda_dual_branch": 0.0, "lambda_cico": 0.0,
            "lambda_sdl": 0.0, "lambda_adl": 0.0, "lambda_hierarchy": 0.35,
            "lambda_calibration": 0.0, "lambda_attribute": 0.0, "lambda_conflict": 0.0,
        },
        "get_adapted": {
            "fusion_mode": "simple",
            "lambda_dual_branch": 1.0, "lambda_cico": 1.0,
            "lambda_sdl": 0.0, "lambda_adl": 0.0, "lambda_hierarchy": 0.0,
            "lambda_calibration": 0.0, "lambda_attribute": 0.0, "lambda_conflict": 0.0,
        },
        "debgcd_adapted": {
            "fusion_mode": "visual",
            "lambda_dual_branch": 0.0, "lambda_cico": 0.0,
            "lambda_sdl": 0.01, "lambda_adl": 1.0, "lambda_hierarchy": 0.0,
            "lambda_calibration": 0.0, "lambda_attribute": 0.0, "lambda_conflict": 0.0,
        },
        "simple_fusion": {
            "fusion_mode": "simple",
            "lambda_dual_branch": 0.0, "lambda_cico": 0.0,
            "lambda_sdl": 0.0, "lambda_adl": 0.0, "lambda_hierarchy": 0.0,
            "lambda_calibration": 0.0, "lambda_attribute": 0.0, "lambda_conflict": 0.0,
        },
        "proposed": {},
    }
    commands = []
    for experiment, overrides in experiments.items():
        for seed in args.seeds:
            config = copy.deepcopy(base)
            config["seed"] = seed
            config["output_dir"] = f"outputs/planogram/{experiment}/seed_{seed}"
            if "fusion_mode" in overrides:
                config["model"]["fusion_mode"] = overrides["fusion_mode"]
            for key, value in overrides.items():
                if key != "fusion_mode":
                    config["train"][key] = value
            path = output / f"{experiment}_seed{seed}.json"
            path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            commands.append(f"retail-gcd --config {path.as_posix()}")
    (output / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"generated {len(commands)} configs and {output / 'commands.txt'}")


if __name__ == "__main__":
    main()

