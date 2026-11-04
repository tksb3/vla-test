"""Step 5: the barrier and the exchange between islands.

Each island is one pipeline process running one instruction from the instruction
space, launched by hand, on its own CARLA port. The islands search independently
but stop at the end of every generation to trade traffic contexts: a context is
worth importing when some other island's control run finished clean in *every*
repetition, because a clean baseline is what makes a collision under instruction
readable as the instruction's doing.

The baseline never sees a prompt, so its behaviour in a given context does not
depend on which island measured it. What differs is which contexts each island
looked at, and that is what the exchange is for.

Two modes:
  --mode start  registers this island and holds until every island has registered
                and the agreed wall-clock start time has passed.
  --mode sync   publishes this generation's results, waits for the other islands,
                then files the baseline-clean contexts they found into this
                island's migration pool for step 02 to draw on.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import read_config, read_json, safe_name, write_json


MODE_START = "start"
MODE_SYNC = "sync"

DEFAULT_POLL_INTERVAL_S = 1.0
DEFAULT_TIMEOUT_S = 43200.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 5: island barrier and migration exchange.")
    parser.add_argument("--mode", choices=(MODE_START, MODE_SYNC), required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="Scenario config; supplies the island block and repetitions.")
    parser.add_argument("--shared-dir", type=Path, required=True,
                        help="Directory every island reads and writes to.")
    parser.add_argument("--island-index", type=int, required=True,
                        help="This island's number, which is also its index into the instruction space.")
    parser.add_argument("--islands", type=int, required=True,
                        help="How many islands take part; the barrier waits for exactly these.")
    parser.add_argument("--start-at", default=None,
                        help="mode start: UTC instant to resume at, e.g. 2026-08-30T09:00:00Z.")
    parser.add_argument("--generation", type=int, default=None,
                        help="mode sync: the generation whose results are being published.")
    parser.add_argument("--population", type=Path, default=None,
                        help="mode sync: step 02's population file, which carries the genotypes.")
    parser.add_argument("--evaluation", type=Path, default=None,
                        help="mode sync: step 04's evaluation for this generation.")
    parser.add_argument("--pool", type=Path, default=None,
                        help="mode sync: this island's migration pool to append to.")
    return parser.parse_args()


def island_name(index: int) -> str:
    return f"island_{index:02d}"


def flags_dir(shared_dir: Path, generation: int | None) -> Path:
    return shared_dir / "flags" / ("start" if generation is None else f"gen_{generation:04d}")


def result_path(shared_dir: Path, index: int, generation: int) -> Path:
    return shared_dir / "results" / island_name(index) / f"gen_{generation:04d}.json"


def island_settings(config: dict[str, Any]) -> tuple[float, float]:
    island = config.get("island", {})
    return (
        float(island.get("poll_interval_s", DEFAULT_POLL_INTERVAL_S)),
        float(island.get("timeout_s", DEFAULT_TIMEOUT_S)),
    )


def raise_flag(directory: Path, index: int) -> Path:
    """Announce that this island's data for the barrier is complete and readable.

    Always written after the data it stands for, so an island that sees the flag
    can read the results without a second round of waiting.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{island_name(index)}.done"
    path.write_text("", encoding="utf-8")
    return path


def wait_for_flags(directory: Path, islands: int, index: int, poll_interval_s: float,
                   timeout_s: float, label: str) -> None:
    expected = {island_name(i) for i in range(islands)} - {island_name(index)}
    deadline = time.monotonic() + timeout_s
    announced: set[str] = set()
    while True:
        present = {path.stem for path in directory.glob("*.done")} & expected
        for name in sorted(present - announced):
            print(f"{label}: {name} is here ({len(present)}/{len(expected)})")
        announced = present
        if present == expected:
            return
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"{label}: timed out after {timeout_s:.0f}s waiting for "
                f"{', '.join(sorted(expected - present))}. Those islands are stopped or stuck; "
                "fix them and restart every island with --resume."
            )
        time.sleep(poll_interval_s)


def wait_until(start_at: str) -> None:
    """Hold until the instant every island was told to resume at.

    The islands are started by hand one after another, so without this the first
    would already be a generation in by the time the last one comes up, and the
    per-generation barrier would just serialise them.
    """
    text = start_at.strip()
    target = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    remaining = (target - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        print(f"start time {target.isoformat()} has already passed; continuing")
        return
    print(f"holding {remaining:.0f}s until {target.isoformat()}")
    time.sleep(remaining)


def collect_entries(population: dict[str, Any], evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    """Join this generation's genotypes to the fitness each repetition produced.

    The genotypes come from the population file rather than ga_archive.json:
    step 02 only archives a generation while evolving past it, which happens
    after this step runs.
    """
    by_case = {
        str(report.get("case_id")): report.get("evaluation") or {}
        for report in evaluation.get("evaluations", [])
    }
    entries = []
    for scene in population.get("candidate_scenes", []):
        candidate_id = scene["candidate_id"]
        report = by_case.get(safe_name(candidate_id))
        if report is None:
            continue
        baseline = report.get("baseline") or {}
        entries.append({
            "candidate_id": candidate_id,
            "individual": scene["individual"],
            "instruction_fitness_by_repetition": report.get("fitness_by_repetition") or {},
            "baseline_fitness_by_repetition": baseline.get("fitness_by_repetition") or {},
            "fitness_differential": report.get("fitness_differential"),
            # Enough of the baseline for an importing island to adopt it wholesale
            # instead of simulating the control run again. classify_outcome in the
            # summarise tools reads entry_status / has_collision /
            # forced_termination, so a baseline that travels without them would
            # come out as 'crashed' wherever it landed.
            "baseline": {
                "status": baseline.get("status"),
                "entry_status": baseline.get("entry_status"),
                "score_composed": baseline.get("score_composed"),
                "has_collision": baseline.get("has_collision"),
                "has_red_light_violation": baseline.get("has_red_light_violation"),
                "has_route_deviation": baseline.get("has_route_deviation"),
                "completed": baseline.get("completed"),
                "forced_termination": baseline.get("forced_termination"),
                "infraction_count": baseline.get("infraction_count"),
                "fitness": baseline.get("fitness"),
                "fitness_values": baseline.get("fitness_values") or [],
                "fitness_by_repetition": baseline.get("fitness_by_repetition") or {},
                "repetitions": baseline.get("repetitions"),
                "usable_repetitions": baseline.get("usable_repetitions"),
                # Set by step 04 when this baseline was itself adopted from a peer
                # rather than measured here; is_baseline_clean refuses to pass
                # those on, so a measurement never travels more than one hop.
                "baseline_source": baseline.get("baseline_source", "measured"),
            },
        })
    return entries


def is_baseline_clean(entry: dict[str, Any], repetitions: int) -> bool:
    """The control run finished with fitness 0 on every repetition, measured here.

    The mean is not enough: a context where the baseline collided once and
    completed twice averages low but is not one where the baseline reliably
    drives clean, which is the only reason to import it.

    A baseline this island adopted from a peer is not passed on either. It would
    otherwise be republished as this island's own, and the next importer would be
    two hops from the only real measurement -- so a context that once happened to
    drive clean could travel from island to island wearing that label forever,
    never tested again. Measured runs here have been seen to break contexts a
    peer had published as clean, which is exactly what the republishing would
    hide.
    """
    baseline = entry.get("baseline") or {}
    if baseline.get("baseline_source") == "reused":
        return False
    by_repetition = entry.get("baseline_fitness_by_repetition") or {}
    if len(by_repetition) != repetitions:
        return False
    return all(float(value) == 0.0 for value in by_repetition.values())


def genotype_key(individual: dict[str, Any]) -> str:
    return json.dumps(individual, sort_keys=True)


def own_genotypes(shared_dir: Path, index: int) -> set[str]:
    """Every genotype this island has already published, across all generations.

    All islands share ga_settings.seed, so generation 0 is the same ten
    individuals everywhere -- importing a peer's copy of a context this island
    already measured would spend a simulation on a known answer. Later
    generations diverge, but the check costs nothing and keeps holding.
    """
    directory = shared_dir / "results" / island_name(index)
    keys: set[str] = set()
    for path in sorted(directory.glob("gen_*.json")):
        for entry in read_json(path).get("entries", []):
            keys.add(genotype_key(entry["individual"]))
    return keys


def merge_into_pool(pool_path: Path, index: int, imported: list[dict[str, Any]]) -> int:
    """Append newly seen contexts, leaving the used flags of existing ones alone."""
    pool = read_json(pool_path) if pool_path.exists() else {
        "step": "05_sync_island",
        "island_index": index,
        "entries": [],
    }
    known = {genotype_key(entry["individual"]) for entry in pool["entries"]}
    added = 0
    for entry in imported:
        key = genotype_key(entry["individual"])
        if key in known:
            continue
        known.add(key)
        pool["entries"].append({**entry, "used": False})
        added += 1
    write_json(pool_path, pool)
    return added


def run_start(args: argparse.Namespace, poll_interval_s: float, timeout_s: float) -> None:
    directory = flags_dir(args.shared_dir, None)
    raise_flag(directory, args.island_index)
    print(f"{island_name(args.island_index)} registered in {directory}")
    if args.islands > 1:
        wait_for_flags(directory, args.islands, args.island_index, poll_interval_s, timeout_s,
                       label="start barrier")
    if args.start_at:
        wait_until(args.start_at)


def run_sync(args: argparse.Namespace, config: dict[str, Any],
             poll_interval_s: float, timeout_s: float) -> None:
    for name in ("generation", "population", "evaluation", "pool"):
        if getattr(args, name) is None:
            raise SystemExit(f"--mode {MODE_SYNC} requires --{name.replace('_', '-')}")

    evaluation = read_json(args.evaluation)
    repetitions = int(evaluation.get("repetitions", 1))
    entries = collect_entries(read_json(args.population), evaluation)

    write_json(result_path(args.shared_dir, args.island_index, args.generation), {
        "step": "05_sync_island",
        "island_index": args.island_index,
        "generation": args.generation,
        "repetitions": repetitions,
        "entries": entries,
    })
    directory = flags_dir(args.shared_dir, args.generation)
    raise_flag(directory, args.island_index)
    print(f"generation {args.generation}: published {len(entries)} candidates")

    if args.islands <= 1:
        return
    wait_for_flags(directory, args.islands, args.island_index, poll_interval_s, timeout_s,
                   label=f"generation {args.generation} barrier")

    seen = own_genotypes(args.shared_dir, args.island_index)
    imported = []
    skipped_own = 0
    for peer in range(args.islands):
        if peer == args.island_index:
            continue
        peer_result = read_json(result_path(args.shared_dir, peer, args.generation))
        peer_repetitions = int(peer_result.get("repetitions", repetitions))
        for entry in peer_result.get("entries", []):
            if not is_baseline_clean(entry, peer_repetitions):
                continue
            if genotype_key(entry["individual"]) in seen:
                skipped_own += 1
                continue
            imported.append({
                "source_island": peer,
                "source_generation": args.generation,
                "candidate_id": entry["candidate_id"],
                "individual": entry["individual"],
                # Carried so step 02 can hand it to step 04 in place of a control
                # run, when ga_settings.reuse_migrant_baseline says to.
                "baseline": entry.get("baseline") or {},
                "baseline_repetitions": peer_repetitions,
                # What the false instruction actually did here, so an importing
                # island can refuse contexts the instruction drove through
                # cleanly (migration_selection 'positive_instruction'). Carried
                # per repetition rather than as a mean, same as the baseline
                # block: the consumer decides how to read it.
                "instruction_fitness_by_repetition":
                    entry.get("instruction_fitness_by_repetition") or {},
                "fitness_differential": entry.get("fitness_differential"),
            })
    added = merge_into_pool(args.pool, args.island_index, imported)
    print(f"generation {args.generation}: {len(imported)} baseline-clean contexts from peers "
          f"({skipped_own} already measured here), {added} new in {args.pool}")


def main() -> None:
    args = parse_args()
    if args.islands < 1:
        raise SystemExit(f"--islands must be >= 1, got {args.islands}")
    if not 0 <= args.island_index < args.islands:
        raise SystemExit(f"--island-index {args.island_index} is outside 0..{args.islands - 1}")

    config = read_config(args.config)
    poll_interval_s, timeout_s = island_settings(config)

    if args.mode == MODE_START:
        run_start(args, poll_interval_s, timeout_s)
    else:
        run_sync(args, config, poll_interval_s, timeout_s)


if __name__ == "__main__":
    main()
