from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import (
    OUTPUT_DIR,
    available_scenarios,
    read_json,
    scenario_input_paths,
)


SRC_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the numbered VLA contradiction-test pipeline.")
    parser.add_argument("--run-simlingo", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Root directory that aggregates every step-01..04 output (one instr_XX subdirectory per instruction).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Number of GA generations per instruction. Defaults to ga_settings.generations in the traffic-context space.",
    )
    parser.add_argument(
        "--instructions",
        type=int,
        default=None,
        help="Number of instruction candidates to iterate. Defaults to all instructions from --instruction-start onward.",
    )
    parser.add_argument(
        "--instruction-start",
        type=int,
        default=0,
        help="0-based index of the first instruction to run. Lets two pipeline processes split the instruction space.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help=(
            "Path to the directory holding one scenario's input set -- common.json, "
            "config.json, instruction_space.json and traffic_context_space.json -- "
            "absolute, or relative to the working directory. "
            f"Scenarios in this repo: {', '.join(available_scenarios()) or '(none)'}."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config JSON. Overrides the one --scenario would pick.",
    )
    parser.add_argument(
        "--instruction-space",
        type=Path,
        default=None,
        help="Instruction-space JSON passed to step 01. Overrides the one --scenario would pick.",
    )
    parser.add_argument(
        "--traffic-context-space",
        type=Path,
        default=None,
        help="Traffic-context space JSON passed to step 02. Overrides the one --scenario would pick.",
    )
    parser.add_argument(
        "--carla-port",
        type=int,
        default=None,
        help="CARLA RPC port passed to step 03 (for parallel CARLA instances).",
    )
    parser.add_argument(
        "--tm-port",
        type=int,
        default=None,
        help="Traffic-manager port passed to step 03 (for parallel CARLA instances).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip work that already has an evaluation file and continue from the first incomplete step.",
    )
    parser.add_argument(
        "--island-index",
        type=int,
        default=None,
        help=(
            "Run as one island of a coordinated set: this process takes instruction number "
            "ISLAND_INDEX and stops at every generation to trade traffic contexts with the "
            "others through --shared-dir. Implies --instruction-start ISLAND_INDEX --instructions 1."
        ),
    )
    parser.add_argument(
        "--shared-dir",
        type=Path,
        default=None,
        help="Island mode: directory every island publishes its per-generation results into.",
    )
    parser.add_argument(
        "--islands",
        type=int,
        default=None,
        help="Island mode: how many islands take part. Defaults to the size of the instruction space.",
    )
    parser.add_argument(
        "--start-at",
        default=None,
        help=(
            "Island mode: UTC instant every island resumes at, e.g. 2026-08-30T09:00:00Z. "
            "The islands are launched by hand, so this is what gets them into the same generation."
        ),
    )
    return verify_island_args(resolve_input_paths(parser.parse_args()))


INPUT_PATH_ARGS = ("config", "instruction_space", "traffic_context_space")


def resolve_input_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Fill config / instruction_space / traffic_context_space.

    --scenario names a directory holding all three; an explicit flag wins over it.
    There is no default input set: every path must end up named one way or the other.
    """
    from_scenario: dict[str, Path] = {}
    if args.scenario is not None:
        try:
            from_scenario = scenario_input_paths(args.scenario)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    for name in INPUT_PATH_ARGS:
        if getattr(args, name) is None and name in from_scenario:
            setattr(args, name, from_scenario[name])

    unset = [f"--{name.replace('_', '-')}" for name in INPUT_PATH_ARGS if getattr(args, name) is None]
    if unset:
        raise SystemExit(
            "missing input paths: " + ", ".join(unset) + " (pass them, or point --scenario at "
            f"a directory holding all three; in this repo: {', '.join(available_scenarios()) or '(none)'})"
        )
    missing = [str(getattr(args, name)) for name in INPUT_PATH_ARGS if not getattr(args, name).is_file()]
    if missing:
        raise SystemExit("input file not found: " + ", ".join(missing))
    return args


ISLAND_ONLY_ARGS = {"shared_dir": "--shared-dir", "islands": "--islands", "start_at": "--start-at"}


def verify_island_args(args: argparse.Namespace) -> argparse.Namespace:
    """Island mode is all-or-nothing, and it owns the instruction selection."""
    if args.island_index is None:
        stray = [flag for name, flag in ISLAND_ONLY_ARGS.items() if getattr(args, name) is not None]
        if stray:
            raise SystemExit(f"{', '.join(stray)} need --island-index")
        return args

    if args.shared_dir is None:
        raise SystemExit("--island-index needs --shared-dir")
    if args.instructions is not None or args.instruction_start:
        raise SystemExit(
            "--instruction-start / --instructions belong to a plain sweep; in island mode "
            "--island-index picks the instruction"
        )
    total = len(read_json(args.instruction_space).get("instructions", []))
    if args.islands is None:
        args.islands = total
    if not 0 <= args.island_index < args.islands:
        raise SystemExit(f"--island-index {args.island_index} is outside 0..{args.islands - 1}")
    if args.islands > total:
        raise SystemExit(
            f"--islands {args.islands} exceeds the {total} instructions in {args.instruction_space}"
        )
    args.instruction_start = args.island_index
    args.instructions = 1
    return args


def run(script: str, extra_args: list[str] | None = None, retries: int = 0) -> None:
    cmd = [sys.executable, str(SRC_DIR / script)]
    if extra_args:
        cmd.extend(extra_args)
    for attempt in range(1, retries + 2):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as exc:
            # The CARLA client can abort the whole step-03 process (SIGABRT from a
            # C++ worker-thread TimeoutException); finished cases are persisted per
            # case, so rerunning the step only redoes the interrupted case.
            if attempt > retries:
                raise
            print(f"{script} failed (attempt {attempt}/{retries + 1}): {exc}; retrying")


class RunLayout:
    """Where one instruction's files live, under the run's --out-dir."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir

    def instruction_dir(self, instruction_index: int) -> Path:
        return self.out_dir / f"instr_{instruction_index:02d}"

    def false_scene(self, instruction_index: int) -> Path:
        return self.instruction_dir(instruction_index) / "for_01_false_scene_search_result.json"

    def population(self, instruction_index: int) -> Path:
        return (self.instruction_dir(instruction_index)
                / "for_02_generated_false_scene_based_traffic_context_search_result.json")

    def archive(self, instruction_index: int) -> Path:
        return self.instruction_dir(instruction_index) / "ga_archive.json"

    def migration_pool(self, instruction_index: int) -> Path:
        return self.instruction_dir(instruction_index) / "migration_pool.json"

    def simulation(self, instruction_index: int, generation: int) -> Path:
        return self.instruction_dir(instruction_index) / "03_carla_simulation" / f"gen_{generation:04d}"

    def evaluation(self, instruction_index: int, generation: int) -> Path:
        return self.instruction_dir(instruction_index) / f"04_evaluation_gen_{generation:04d}.json"


def step2_args(args: argparse.Namespace, layout: RunLayout, instruction_index: int) -> list[str]:
    arguments = [
        "--traffic-context-space", str(args.traffic_context_space),
        "--false-scene", str(layout.false_scene(instruction_index)),
        "--out", str(layout.population(instruction_index)),
        "--archive", str(layout.archive(instruction_index)),
    ]
    if args.island_index is not None:
        arguments += ["--migration-pool", str(layout.migration_pool(instruction_index))]
    return arguments


def island_args(args: argparse.Namespace) -> list[str]:
    return [
        "--config", str(args.config),
        "--shared-dir", str(args.shared_dir),
        "--island-index", str(args.island_index),
        "--islands", str(args.islands),
    ]


def sync_island(args: argparse.Namespace, layout: RunLayout, instruction_index: int,
                generation: int) -> None:
    """Publish this generation and wait for the other islands to do the same."""
    run("05_sync_island.py", island_args(args) + [
        "--mode", "sync",
        "--generation", str(generation),
        "--population", str(layout.population(instruction_index)),
        "--evaluation", str(layout.evaluation(instruction_index, generation)),
        "--pool", str(layout.migration_pool(instruction_index)),
    ])


def start_instruction(args: argparse.Namespace, layout: RunLayout, instruction_index: int) -> None:
    """Pick the instruction's false scene and sample its generation-0 population."""
    run(
        "01_search_false_scene.py",
        [
            "--config", str(args.config),
            "--instruction-space", str(args.instruction_space),
            "--instruction-index", str(instruction_index),
            "--out", str(layout.false_scene(instruction_index)),
        ],
    )
    run("02_search_traffic_context.py", step2_args(args, layout, instruction_index))


def advance_generation(
    args: argparse.Namespace,
    layout: RunLayout,
    instruction_index: int,
    generation: int,
    evolve: bool,
) -> None:
    """Simulate and evaluate one generation, then fold its scores back into the GA.

    `evolve` is False for a generation with no successor: the scores still have to
    reach the archive, but advancing the GA past the run's last generation would
    leave a population nobody ever simulates.
    """
    step3_args = [
        "--config", str(args.config),
        "--scene", str(layout.population(instruction_index)),
        "--out-dir", str(layout.simulation(instruction_index, generation)),
        "--all-candidates",
    ]
    if args.run_simlingo:
        step3_args.append("--run-simlingo")
    if args.carla_port is not None:
        step3_args.extend(["--carla-port", str(args.carla_port)])
    if args.tm_port is not None:
        step3_args.extend(["--tm-port", str(args.tm_port)])
    run("03_run_carla_simulation.py", step3_args, retries=2)

    run(
        "04_evaluate.py",
        [
            "--simulation", str(layout.simulation(instruction_index, generation) / "simulation_batch_summary.json"),
            "--out", str(layout.evaluation(instruction_index, generation)),
        ],
    )

    # The exchange sits between the evaluation and the evolve step: step 02 draws
    # on the pool this fills, and every island has to have published before any of
    # them reads.
    if args.island_index is not None:
        sync_island(args, layout, instruction_index, generation)

    step2 = ["--evaluation", str(layout.evaluation(instruction_index, generation))] + step2_args(
        args, layout, instruction_index
    )
    run("02_search_traffic_context.py", step2 if evolve else step2 + ["--archive-only"])


def run_instruction_sweep(args: argparse.Namespace, layout: RunLayout, total_instructions: int) -> None:
    generations = args.generations
    if generations is None:
        space = read_json(args.traffic_context_space)
        generations = int(space.get("ga_settings", {}).get("generations", 1))

    first_instruction = args.instruction_start
    instruction_count = args.instructions
    if instruction_count is None:
        instruction_count = total_instructions - first_instruction
    end_instruction = first_instruction + instruction_count
    if first_instruction < 0 or instruction_count <= 0 or end_instruction > total_instructions:
        print(
            f"no instruction candidates to run (start={first_instruction}, count={instruction_count}, "
            f"instruction space has {total_instructions})"
        )
        return

    start_instruction_index = first_instruction
    start_generation = 0
    if args.resume:
        while start_instruction_index < end_instruction:
            start_generation = 0
            while (start_generation < generations
                   and layout.evaluation(start_instruction_index, start_generation).exists()):
                start_generation += 1
            if start_generation < generations:
                break
            start_instruction_index += 1
        if start_instruction_index >= end_instruction:
            print(f"all {instruction_count} instructions x {generations} generations are already evaluated; nothing to resume")
            return
        if start_instruction_index > first_instruction or start_generation > 0:
            print(f"resuming from instruction {start_instruction_index}, generation {start_generation}")

    for instruction_index in range(start_instruction_index, end_instruction):
        first_generation = start_generation if instruction_index == start_instruction_index else 0

        if first_generation == 0:
            start_instruction(args, layout, instruction_index)
        else:
            # If the crash happened after step 04 but before the evolve step, the
            # population file is still at the previous generation; evolve it first.
            population = read_json(layout.population(instruction_index))
            if int(population.get("generation", -1)) < first_generation:
                run(
                    "02_search_traffic_context.py",
                    ["--evaluation", str(layout.evaluation(instruction_index, first_generation - 1))]
                    + step2_args(args, layout, instruction_index),
                )

        for generation in range(first_generation, generations):
            advance_generation(args, layout, instruction_index, generation, evolve=generation < generations - 1)


def main() -> None:
    args = parse_args()
    layout = RunLayout(args.out_dir)
    total_instructions = len(read_json(args.instruction_space).get("instructions", []))
    if args.island_index is not None and not args.resume:
        # A resumed island rejoins at the next generation barrier, so holding it
        # here would only make the others wait for a start that already happened.
        run("05_sync_island.py", island_args(args) + ["--mode", "start"]
            + (["--start-at", args.start_at] if args.start_at else []))
    run_instruction_sweep(args, layout, total_instructions)


if __name__ == "__main__":
    main()
