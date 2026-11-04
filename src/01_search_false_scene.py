from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    parse_route_xml,
    read_config,
    read_json,
    safe_name,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1: build the false scene for one instruction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--instruction-space", type=Path, required=True)
    parser.add_argument(
        "--instruction-index",
        type=int,
        required=True,
        help="0-based index into the instruction space; the pipeline walks the space one index at a time.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    instruction_space = read_json(args.instruction_space)
    route_xml = Path(config["fixed_route_xml"])
    route = parse_route_xml(route_xml)

    instructions = instruction_space["instructions"]
    if not 0 <= args.instruction_index < len(instructions):
        raise SystemExit(
            f"--instruction-index {args.instruction_index} is out of range "
            f"(instruction space has {len(instructions)} instructions)"
        )
    instruction = instructions[args.instruction_index]

    runtime_prompt_tag = instruction_space.get("runtime_prompt_tag", "instruction_following")
    route_base_id = safe_name(f"{route.get('town') or config['fixed_map']}_{route.get('route_id') or route_xml.stem}")
    best_candidate = {
        # The 1-based suffix predates --instruction-index and is kept because it is
        # baked into every candidate_id of every finished experiment.
        "candidate_id": f"{route_base_id}_false_scene_{args.instruction_index + 1:02d}",
        "town": config["fixed_map"],
        "route_xml": str(route_xml),
        "traffic_scenario": config["fixed_traffic_scenario"],
        "instruction": instruction,
        "runtime_prompt_tag": runtime_prompt_tag,
    }
    write_json(
        args.out,
        {
            "step": "01_search_false_scene",
            "description": (
                "Pins one instruction from the instruction space to the route that step 02 will "
                "then search traffic contexts for. Nothing is rendered here: the contradiction "
                "lives entirely in the instruction text, set against whatever the route actually "
                "presents, and instruction_space.json's own description says which is which."
            ),
            "instruction_space": str(args.instruction_space),
            "instruction_index": args.instruction_index,
            "best_candidate": best_candidate,
        },
    )
    print(f"wrote {args.out} (instruction_index={args.instruction_index}, instruction={instruction!r})")


if __name__ == "__main__":
    main()
