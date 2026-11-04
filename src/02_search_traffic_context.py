from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    OUTPUT_DIR,
    read_json,
    safe_name,
    write_json,
)
from ga_pymoo import (
    INFRASTRUCTURE_VIOLATION,
    MIGRATION_SELECTION_NEAREST,
    MIGRATION_SELECTION_POSITIVE,
    SELECTION_DIFFERENTIAL,
    TrafficContextProblem,
    append_individuals,
    ask_offspring,
    decode_individual,
    distance_to_population,
    inactive_migration_settings,
    is_constrained,
    migration_selection_mode,
    load_state,
    make_algorithm,
    make_config_snapshot,
    operator_parameters_mode,
    parse_constraints,
    reuse_migrant_baseline,
    selection_objective_mode,
    save_state,
    unused_ga_settings,
    verify_config_unchanged,
    verify_repetitions_unchanged,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 2: GA-search traffic context for the false scene.")
    parser.add_argument("--traffic-context-space", type=Path, required=True)
    parser.add_argument("--false-scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, default=None, help="Step 04 evaluation JSON. When given, evolve the next generation instead of sampling generation 0.")
    parser.add_argument(
        "--archive-only",
        action="store_true",
        help=(
            "Score --evaluation into --archive and stop: no tell(), no ask(), no new population. "
            "Used for the last generation, which has no successor to evolve into."
        ),
    )
    parser.add_argument("--previous", type=Path, default=None, help="Previous generation output of this script (defaults to --out).")
    parser.add_argument("--archive", type=Path, default=OUTPUT_DIR / "ga_archive.json", help="Archive of evaluated individuals across generations.")
    parser.add_argument(
        "--ga-state",
        type=Path,
        default=None,
        help="Pickled pymoo algorithm state carried between generations (defaults to ga_state.pkl next to --archive).",
    )
    parser.add_argument(
        "--migration-pool",
        type=Path,
        default=None,
        help=(
            "Island mode: pool of baseline-clean contexts filed by 05_sync_island.py. "
            "ga_settings.migration_count of the offspring are replaced with entries from it."
        ),
    )
    return parser.parse_args()


# The CARLA weather fields step 03 writes into the route XML, in the order the
# <weather> attributes read most naturally. route_percentage is deliberately not
# here: common.py pins it to 0 and 100 with identical values so the weather stays
# constant along the route, which keeps it from confounding the differential.
WEATHER_GENES = (
    "cloudiness",
    "fog_density",
    "precipitation",
    "precipitation_deposits",
    "wetness",
    "wind_intensity",
    "sun_altitude_angle",
    "sun_azimuth_angle",
)


def to_weather(individual: dict[str, Any]) -> dict[str, float]:
    """The weather block for one individual.

    Every CARLA weather field is its own float gene. An older scheme searched a
    named profile plus an intensity and interpolated between the profile's two
    endpoints; it is gone, so a search space carrying weather_profile and
    weather_intensity instead of these eight now fails here rather than silently
    producing the wrong weather.
    """
    missing = [name for name in WEATHER_GENES if name not in individual]
    if missing:
        raise ValueError(
            f"weather genes missing from the individual: {missing}; ga_search_space has to "
            f"define all of {list(WEATHER_GENES)} as floats"
        )
    return {name: round(float(individual[name]), 3) for name in WEATHER_GENES}


def find_scenario_definition(space: dict[str, Any], scenario_type: str) -> dict[str, Any]:
    for scenario in space.get("traffic_scenarios", []):
        if scenario.get("scenario_type") == scenario_type:
            return scenario
    raise ValueError(f"no GA search space for traffic_scenario={scenario_type}")


def to_signalized_junction_left_turn_parameters(individual: dict[str, Any]) -> dict[str, Any]:
    return {
        "flow_speed": individual["oncoming_vehicle_speed_mps"],
        "source_dist_interval": {
            "from": individual["oncoming_vehicle_spawn_interval_from_m"],
            "to": individual["oncoming_vehicle_spawn_interval_to_m"],
        },
    }


def to_enter_actor_flow_parameters(individual: dict[str, Any]) -> dict[str, Any]:
    return {
        "flow_speed": individual["flow_speed_mps"],
        "source_dist_interval": {
            "from": individual["flow_spawn_interval_from_m"],
            "to": individual["flow_spawn_interval_to_m"],
        },
    }


def to_parked_obstacle_parameters(individual: dict[str, Any]) -> dict[str, Any]:
    return {
        "distance": individual["obstacle_distance_m"],
        "speed": individual["background_max_speed_kph"],
    }


def to_parked_obstacle_two_ways_parameters(individual: dict[str, Any]) -> dict[str, Any]:
    interval = individual["opposite_flow_frequency_m"]
    return {
        "distance": individual["obstacle_distance_m"],
        "frequency": {"from": interval["from"], "to": interval["to"]},
    }


# traffic_scenario -> function mapping a GA individual onto that scenario's
# scenario_parameters. Adding a scenario means adding one adapter above plus one
# line here; the gene names it reads must match the scenario's ga_search_space.
SCENARIO_PARAMETER_ADAPTERS = {
    "SignalizedJunctionLeftTurn": to_signalized_junction_left_turn_parameters,
    # Same genes as the signalized version; only the scenario type in the XML differs.
    "NonSignalizedJunctionLeftTurn": to_signalized_junction_left_turn_parameters,
    "EnterActorFlow": to_enter_actor_flow_parameters,
    "ParkedObstacle": to_parked_obstacle_parameters,
    # Same genes as ParkedObstacle; only the scenario type in the XML differs.
    "ConstructionObstacle": to_parked_obstacle_parameters,
    "ParkedObstacleTwoWays": to_parked_obstacle_two_ways_parameters,
}


def to_runtime_parameters(traffic_scenario: str, individual: dict[str, Any]) -> dict[str, Any]:
    adapter = SCENARIO_PARAMETER_ADAPTERS.get(traffic_scenario)
    if adapter is None:
        raise ValueError(
            f"unsupported traffic_scenario for Step 02 GA adapter: {traffic_scenario} "
            f"(registered: {', '.join(sorted(SCENARIO_PARAMETER_ADAPTERS))})"
        )
    try:
        scenario_parameters = adapter(individual)
    except KeyError as exc:
        raise ValueError(
            f"{traffic_scenario} adapter needs gene {exc} but the individual only has "
            f"{sorted(individual)}; check ga_search_space in the traffic-context space"
        ) from exc
    return {
        "weather": to_weather(individual),
        "scenario_parameters": scenario_parameters,
    }


# Genes that steer the pipeline rather than the route XML, keyed by the block
# step 03 reads them from. They stay out of scenario_runtime_parameters because
# that block is written into the route XML, and these have no place there.
INSTRUCTION_TRIGGER_GENES = {"instruction_fire_distance_m": "fire_distance_m"}


def to_instruction_trigger(individual: dict[str, Any]) -> dict[str, Any] | None:
    """Per-candidate overrides for config.instruction_trigger, or None."""
    overrides = {
        key: individual[gene]
        for gene, key in INSTRUCTION_TRIGGER_GENES.items()
        if gene in individual
    }
    return overrides or None


def to_instruction_trigger_bounds(search_space: dict[str, Any]) -> dict[str, Any] | None:
    """The searched ceiling for each trigger override.

    Step 03 validates the trigger against the route, but a candidate only trips
    an impossible bound when it happens to sample near the top of it -- with ten
    candidates a generation that can take several generations to surface. Passing
    the ceiling along lets step 03 reject the bound itself, before generation 0
    simulates anything.
    """
    bounds = {
        key: float(search_space[gene]["max"])
        for gene, key in INSTRUCTION_TRIGGER_GENES.items()
        if gene in search_space and "max" in search_space[gene]
    }
    return bounds or None


def pooled_instruction_fitness(entry: dict[str, Any]) -> float | None:
    """Mean instruction fitness recorded on a pooled context, or None if absent.

    Pools written before 05_sync_island carried this do not have it, and a
    context whose genes predate a search-space change can be in the same file.
    None means 'not known', which positive_instruction treats as not positive
    rather than guessing.
    """
    by_repetition = entry.get("instruction_fitness_by_repetition") or {}
    if not by_repetition:
        return None
    values = [float(value) for value in by_repetition.values()]
    return sum(values) / len(values)


def positive_instruction_only(
    available: list[dict[str, Any]], pooled: int
) -> list[dict[str, Any]]:
    """Pooled contexts where the false instruction itself scored above zero.

    Everything in the pool is baseline-clean -- 05_sync_island only publishes
    contexts whose control run was 0 on every repetition -- so for these entries
    the instruction mean and fitness_differential are the same number, and
    filtering on either gives the same shortlist.
    """
    scored = [(pooled_instruction_fitness(entry), entry) for entry in available]
    known = [entry for fitness, entry in scored if fitness is not None]
    kept = [entry for fitness, entry in scored if fitness is not None and fitness > 0.0]
    print(f"migration: {pooled} pooled, {len(available)} unused, "
          f"{len(known)} with instruction fitness recorded, {len(kept)} positive")
    return kept


def select_migrants(
    available: list[dict[str, Any]],
    take: int,
    algorithm: Any,
    search_space: dict[str, Any],
    selection: str,
) -> list[tuple[dict[str, Any], float | None]]:
    """Which pooled contexts to claim, paired with the distance that picked them.

    'nearest' ranks the pool by how close each context sits to the island's own
    surviving population, so the imports land in the region this island is
    already working in rather than wherever a peer happened to look. 'random'
    and 'positive_instruction' both draw uniformly -- the latter's filter has
    already been applied to `available` by draw_migrants, because the shortlist
    has to exist before `take` can be sized against it.
    """
    if selection != MIGRATION_SELECTION_NEAREST:
        drawn = algorithm.random_state.choice(len(available), size=take, replace=False)
        return [(available[int(position)], None) for position in drawn]

    scored: list[tuple[float, dict[str, Any]]] = []
    for entry in available:
        try:
            distance = distance_to_population(entry["individual"], algorithm.pop, search_space)
        except KeyError:
            # A pooled context whose genes predate a search-space change. Skipping
            # it beats crashing the generation over one stale pool entry.
            continue
        scored.append((distance, entry))
    if not scored:
        print("migration: no pooled context matches the current search space")
        return []

    # Shuffled before the stable sort so equal distances do not always resolve to
    # whichever island happened to publish first.
    algorithm.random_state.shuffle(scored)
    scored.sort(key=lambda pair: pair[0])
    return [(entry, distance) for distance, entry in scored[:take]]


# Which offspring give up their slot to an imported context.
def draw_migrants(
    pool_path: Path | None,
    algorithm: Any,
    ga_settings: dict[str, Any],
    search_space: dict[str, Any],
) -> list[tuple[dict[str, Any], float | None]]:
    """Claim up to migration_count contexts other islands found baseline-clean.

    Drawn before ask() so the island can mate exactly that many fewer offspring:
    the imported contexts take the free slots instead of displacing children that
    were bred only to be thrown away. The claim is recorded in the pool right
    here, so a context is never handed to two generations.
    """
    count = int(ga_settings.get("migration_count", 0))
    if count <= 0 or pool_path is None or not pool_path.exists():
        return []

    pool = read_json(pool_path)
    entries = pool.get("entries", [])
    available = [entry for entry in entries if not entry.get("used")]
    if not available:
        print(f"migration: nothing to draw ({len(entries)} pooled, all already used)"
              if entries else "migration: the pool is still empty")
        return []

    selection = migration_selection_mode(ga_settings)
    # Applied before take is sized, not inside select_migrants: the shortlist can
    # be smaller than migration_count, and take has to be capped against the
    # shortlist rather than against the whole pool.
    if selection == MIGRATION_SELECTION_POSITIVE:
        available = positive_instruction_only(available, len(entries))
        if not available:
            # Deliberately no fallback to a uniform draw. Falling back would mix
            # generations that imported on the stated rule with generations that
            # imported on a different one, under the same label in the log, and
            # nothing downstream could tell them apart. Returning nothing lets
            # ask_offspring breed the full population instead.
            return []

    # One short of pop_size at most: a generation made entirely of imports would
    # stop being this island's own search.
    take = min(count, len(available), int(algorithm.pop_size) - 1)
    if take <= 0:
        return []

    # Drawing without replacement, and the used flag keeps a context from coming
    # back in a later generation.
    sources = select_migrants(available, take, algorithm, search_space, selection)
    if not sources:
        return []
    for entry, _ in sources:
        entry["used"] = True
    write_json(pool_path, pool)

    print(f"migration: importing {len(sources)} contexts ({selection})")
    for entry, distance in sources:
        origin = f"island {entry['source_island']} gen {entry['source_generation']}"
        print(f"  {origin}" if distance is None else f"  {origin}  distance {distance:.3f}")
    return sources


def baseline_reuse_payload(
    entry: dict[str, Any],
    repetitions: int | None,
    enabled: bool,
) -> dict[str, Any] | None:
    """The source island's baseline, for step 03/04 to use instead of rerunning it.

    None when reuse is off or the pool entry predates the carried measurement, in
    which case the control run happens as usual.

    A repetition-count mismatch is fatal rather than a quiet fallback: every
    island reads the same scenario config, and run_pipeline has no --repetitions
    of its own, so the counts can only differ if the islands were launched
    against different inputs. Averaging a 3-run baseline against a 5-run
    instruction would silently change what the differential means.
    """
    if not enabled:
        return None
    baseline = entry.get("baseline") or {}
    if not baseline.get("fitness_by_repetition"):
        return None

    carried = entry.get("baseline_repetitions")
    if repetitions is not None and carried is not None and int(carried) != int(repetitions):
        raise SystemExit(
            f"migrant {entry.get('candidate_id')} from island {entry.get('source_island')} "
            f"carries {carried} baseline repetitions but this island runs {repetitions}; "
            "every island reads the same scenario config, so this means the islands were "
            "launched against different inputs -- fix that rather than mixing baselines "
            "measured over different numbers of runs"
        )
    return {
        "source_island": entry.get("source_island"),
        "source_generation": entry.get("source_generation"),
        "source_candidate_id": entry.get("candidate_id"),
        "repetitions": carried,
        "evaluation": baseline,
    }


def migration_origin(entry: dict[str, Any], distance: float | None) -> dict[str, Any]:
    """Provenance recorded on a candidate scene that came from another island.

    distance is the normalized gene-space gap to this island's population that
    ranked the context, or None when it was drawn at random. Keeping it means a
    later pass can ask whether near imports actually outperformed far ones.
    """
    return {
        "type": "migration",
        "source_island": entry.get("source_island"),
        "source_generation": entry.get("source_generation"),
        "source_candidate_id": entry.get("candidate_id"),
        "distance": None if distance is None else round(distance, 4),
    }


def with_baseline_reuse(origin: dict[str, Any], reuse: dict[str, Any] | None) -> dict[str, Any]:
    """Record on the origin whether this candidate skipped its control run."""
    return {**origin, "baseline_reused": reuse is not None}


def make_candidate_scene(
    false_scene: dict[str, Any],
    individual: dict[str, Any],
    candidate_index: int,
    generation: int,
    origin: dict[str, Any],
    search_space: dict[str, Any],
    baseline_reuse: dict[str, Any] | None = None,
) -> dict[str, Any]:
    traffic_scenario = false_scene["traffic_scenario"]
    scene = {
        "candidate_id": f"{false_scene['candidate_id']}_gen{generation:02d}_traffic_{candidate_index:04d}",
        "generation": generation,
        "individual": individual,
        "origin": origin,
        "traffic_scenario": traffic_scenario,
        "route_xml": false_scene["route_xml"],
        "instruction": false_scene["instruction"],
        "runtime_prompt_tag": false_scene.get("runtime_prompt_tag"),
        "scenario_runtime_parameters": to_runtime_parameters(traffic_scenario, individual),
    }
    # Only present when the search space declares the gene, so a scenario that
    # does not search the trigger keeps the exact scene shape it had before.
    instruction_trigger = to_instruction_trigger(individual)
    if instruction_trigger is not None:
        scene["instruction_trigger"] = instruction_trigger
        scene["instruction_trigger_bounds"] = to_instruction_trigger_bounds(search_space)
    # Only on candidates that skip the control run, so a scene without it keeps
    # the exact shape it had before. Step 03 reads it to leave the baseline role
    # unrun; step 04 reads it to score the differential against it anyway.
    if baseline_reuse is not None:
        scene["baseline_reuse"] = baseline_reuse
    return scene


# Selection-fitness penalty for cases with no usable evaluation: missing data,
# or a genuine infrastructure failure (CARLA crash etc., not the genes' fault).
# Forced termination (stall guard / timeout) is NOT penalized here anymore --
# fitness_differential already reflects it, since step 03's SIGINT-based
# shutdown lets the leaderboard finish writing collision data even on a
# stall-guard kill.
FORCED_TERMINATION_PENALTY = -1000.0

# Cases that failed for infrastructure reasons (CARLA crash etc.), not because of
# the genes. They rank worst for selection, but are NOT treated as evaluated for
# dedup purposes, so the same genes may be simulated again in a later generation.
INFRASTRUCTURE_FAILURE_STATUSES = {"evaluation_failed", "terminated_carla_unreachable"}


def load_evaluations_by_case(evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reports = evaluation["evaluations"] if "evaluations" in evaluation else [evaluation]
    by_case = {}
    for report in reports:
        case_id = report.get("case_id")
        if case_id is not None:
            by_case[case_id] = report["evaluation"]
    return by_case


def selection_fitness(case_evaluation: dict[str, Any] | None) -> float:
    if case_evaluation is None:
        return FORCED_TERMINATION_PENALTY
    if case_evaluation.get("status") in INFRASTRUCTURE_FAILURE_STATUSES:
        return FORCED_TERMINATION_PENALTY
    if "baseline" in case_evaluation:
        # Differential mode (step 04 embedded the control run). Whether either
        # role froze no longer matters here: step 03's SIGINT-based shutdown
        # (see kill_process_group) lets the leaderboard finish writing collision
        # data even on a stall-guard kill, so fitness_differential already
        # reflects what happened -- forced_termination no longer needs a
        # separate penalty path.
        baseline = case_evaluation.get("baseline")
        if baseline is None:
            return FORCED_TERMINATION_PENALTY
        if baseline.get("status") in INFRASTRUCTURE_FAILURE_STATUSES:
            return FORCED_TERMINATION_PENALTY
        return float(case_evaluation["fitness_differential"])
    return float(case_evaluation["fitness"])


def baseline_collision_violation(case_evaluation: dict[str, Any] | None) -> float:
    """How much of the control run collided, as the constraint pymoo ranks on.

    0 means the baseline never collided over its repetitions, which is the only
    state the constraint accepts; pymoo then puts the candidate ahead of every
    infeasible one regardless of objective. A partly-collided baseline keeps its
    share (1/3, 2/3, 1) so infeasible candidates still order among themselves
    rather than all tying.

    A run with no usable evaluation gets INFRASTRUCTURE_VIOLATION, above
    anything collisions can produce: nothing is known about its genes, so it
    should not outrank a candidate that was actually measured.
    """
    if case_evaluation is None:
        return INFRASTRUCTURE_VIOLATION
    if case_evaluation.get("status") in INFRASTRUCTURE_FAILURE_STATUSES:
        return INFRASTRUCTURE_VIOLATION
    baseline = case_evaluation.get("baseline")
    if baseline is None:
        return INFRASTRUCTURE_VIOLATION
    if baseline.get("status") in INFRASTRUCTURE_FAILURE_STATUSES:
        return INFRASTRUCTURE_VIOLATION
    usable = int(baseline.get("usable_repetitions") or 0)
    if usable <= 0:
        return INFRASTRUCTURE_VIOLATION
    return int(baseline.get("has_collision_count") or 0) / usable


def instruction_objective(case_evaluation: dict[str, Any] | None) -> float:
    """The instruction run's own fitness, which constraint mode maximises.

    Nothing is subtracted: the control run is handled by the constraint, so its
    variance stays out of the number being climbed. An unmeasured run scores 0
    and is ruled out by its violation instead.
    """
    if case_evaluation is None:
        return 0.0
    if case_evaluation.get("status") in INFRASTRUCTURE_FAILURE_STATUSES:
        return 0.0
    return float(case_evaluation.get("fitness") or 0.0)


def objective_and_violation(
    case_evaluation: dict[str, Any] | None,
    mode: str,
) -> tuple[float, float]:
    """The value selection maximises, paired with the constraint violation.

    Differential mode reports violation 0 throughout: its problem declares no
    constraint, so cv stays 0 and selection runs on the objective alone, exactly
    as it did before this split existed.
    """
    if mode == SELECTION_DIFFERENTIAL:
        return selection_fitness(case_evaluation), 0.0
    return instruction_objective(case_evaluation), baseline_collision_violation(case_evaluation)


def better_for_selection(entry: dict[str, Any]) -> tuple[float, float]:
    """Sort key matching how pymoo ranks: feasible first, then objective.

    Used for the `best` line in the log, which would otherwise call an infeasible
    candidate the generation's best purely because its objective was large.
    """
    return (float(entry.get("selection_violation") or 0.0), -float(entry["selection_fitness"]))


def score_previous_population(previous: dict[str, Any], evaluation: dict[str, Any],
                              mode: str = SELECTION_DIFFERENTIAL) -> list[dict[str, Any]]:
    evaluations_by_case = load_evaluations_by_case(evaluation)
    scored = []
    for scene in previous.get("candidate_scenes", []):
        if "individual" not in scene:
            raise ValueError(
                f"candidate {scene.get('candidate_id')} has no individual (genotype); "
                "regenerate generation 0 with this script before evolving"
            )
        # Step 03 derives case_id via safe_name(candidate_id).
        case_id = safe_name(scene["candidate_id"])
        case_evaluation = evaluations_by_case.get(case_id)
        baseline_evaluation = case_evaluation.get("baseline") if case_evaluation else None
        infrastructure_failure = (
            case_evaluation is None
            or case_evaluation.get("status") in INFRASTRUCTURE_FAILURE_STATUSES
            or (
                "baseline" in case_evaluation
                and (
                    baseline_evaluation is None
                    or baseline_evaluation.get("status") in INFRASTRUCTURE_FAILURE_STATUSES
                )
            )
        )
        entry = {
            "candidate_id": scene["candidate_id"],
            "individual": scene["individual"],
            # Carried into the archive so a later pass can tell an import from a
            # bred offspring, and read the distance that picked it, without
            # re-deriving provenance by matching genotypes against the pool.
            "origin": scene.get("origin"),
            "fitness": float(case_evaluation["fitness"]) if case_evaluation else 0.0,
            "forced_termination": bool(case_evaluation.get("forced_termination")) if case_evaluation else None,
            "infrastructure_failure": infrastructure_failure,
            "evaluated": case_evaluation is not None,
        }
        # selection_fitness is whatever the configured mode maximises, and
        # selection_violation is 0 unless the mode declares a constraint. Both are
        # archived so a later pass can replay the ranking without re-deriving it.
        objective, violation = objective_and_violation(case_evaluation, mode)
        entry["selection_fitness"] = objective
        entry["selection_violation"] = violation
        if case_evaluation and "fitness_differential" in case_evaluation:
            entry["fitness_differential"] = float(case_evaluation["fitness_differential"])
        scored.append(entry)
    if not scored:
        raise ValueError("previous generation has no candidate_scenes")
    return scored


def append_to_archive(archive_path: Path, scored: list[dict[str, Any]], generation: int) -> dict[str, Any]:
    archive = read_json(archive_path) if archive_path.exists() else {"entries": []}
    known = {(entry["candidate_id"], entry["generation"]) for entry in archive["entries"]}
    for entry in scored:
        if (entry["candidate_id"], generation) not in known:
            archive_entry = {
                "generation": generation,
                "candidate_id": entry["candidate_id"],
                "individual": entry["individual"],
                "origin": entry.get("origin"),
                "fitness": entry["fitness"],
                "selection_fitness": entry["selection_fitness"],
                "selection_violation": entry.get("selection_violation"),
                "forced_termination": entry["forced_termination"],
                "infrastructure_failure": entry["infrastructure_failure"],
                "evaluated": entry["evaluated"],
            }
            if "fitness_differential" in entry:
                archive_entry["fitness_differential"] = entry["fitness_differential"]
            archive["entries"].append(archive_entry)
    write_json(archive_path, archive)
    return archive


def main() -> None:
    args = parse_args()
    traffic_context_space = read_json(args.traffic_context_space)
    false_scene = read_json(args.false_scene)["best_candidate"]
    scenario = find_scenario_definition(traffic_context_space, false_scene["traffic_scenario"])
    search_space = scenario["ga_search_space"]
    gene_constraints = scenario.get("gene_constraints", [])
    constraints = parse_constraints(scenario)
    ga_settings = traffic_context_space.get("ga_settings", {})
    base_seed = int(ga_settings.get("seed", 1))
    objective_mode = selection_objective_mode(ga_settings)
    ga_state_path = args.ga_state or args.archive.with_name("ga_state.pkl")

    if args.archive_only:
        # The last generation is evaluated but never evolved, so nothing would
        # otherwise fold it into the archive. Score it and stop: touching the GA
        # state here would advance it past the run's final generation.
        if args.evaluation is None:
            raise SystemExit("--archive-only requires --evaluation")
        previous = read_json(args.previous or args.out)
        generation = int(previous.get("generation", 0))
        scored = score_previous_population(previous, read_json(args.evaluation), objective_mode)
        append_to_archive(args.archive, scored, generation)
        best = min(scored, key=better_for_selection)
        print(
            f"generation {generation} best: {best['candidate_id']} "
            f"fitness={best['fitness']} selection_fitness={best['selection_fitness']}"
            f" violation={best['selection_violation']} (archived, not evolved)"
        )
        return

    unused = unused_ga_settings(ga_settings)
    print(f"operator_parameters={operator_parameters_mode(ga_settings)}", end="")
    print(f" (unused ga_settings: {', '.join(unused)})" if unused else "")

    # Whether anything is actually imported this run: a plain sweep gets no pool,
    # and an island told migration_count 0 has one but draws from it. Reported so
    # the log says which of the two it is, rather than leaving a scenario file
    # that still carries migration keys looking like it is exchanging contexts.
    migrating = args.migration_pool is not None and int(ga_settings.get("migration_count", 0)) > 0
    inactive = inactive_migration_settings(ga_settings, migrating)
    print(f"migration={'on' if migrating else 'off'}", end="")
    print(f" (unused ga_settings: {', '.join(inactive)})" if inactive else "")
    print(f"selection_objective={objective_mode}")

    snapshot = make_config_snapshot(search_space, ga_settings, gene_constraints)

    if args.evaluation is not None:
        previous = read_json(args.previous or args.out)
        evaluation = read_json(args.evaluation)
        generation = int(previous.get("generation", 0)) + 1
        state = load_state(ga_state_path)
        verify_config_unchanged(state, search_space, ga_settings, gene_constraints)
        repetitions = int(evaluation.get("repetitions", 1))
        verify_repetitions_unchanged(state, repetitions)
        algorithm = state["algorithm"]
        if int(state.get("generation", -1)) >= generation:
            # An earlier run already advanced the GA to this generation but died
            # before rewriting the population file. Unlike the previous stateless
            # GA, telling the algorithm a second time would corrupt its state, so
            # re-emit the population the stored state already holds. Redrawing
            # migrants is not an option either: the pool already marked them used.
            pending = state["pending"]
            origins = state.get("origins") or [{"type": "offspring"}] * len(pending)
            reuses = state.get("baseline_reuses") or [None] * len(pending)
            print(f"GA state is already at generation {state['generation']}; reusing its population")
        else:
            pending = state["pending"]
            scored = score_previous_population(previous, evaluation, objective_mode)
            if len(scored) != len(pending):
                raise SystemExit(
                    f"{args.evaluation} scores {len(scored)} candidates but the GA state expects "
                    f"{len(pending)}; regenerate generation 0 before evolving"
                )
            # pymoo minimises, the pipeline maximises, so the objective is negated.
            # Both arrays, not lists: calc_cv reads .ndim off whatever G holds.
            pending.set("F", np.array([[-entry["selection_fitness"]] for entry in scored]))
            if is_constrained(ga_settings):
                pending.set("G", np.array([[entry["selection_violation"]] for entry in scored]))
            algorithm.tell(infills=pending)
            append_to_archive(args.archive, scored, generation - 1)
            best = min(scored, key=better_for_selection)
            feasible = sum(1 for entry in scored if not entry["selection_violation"])
            print(
                f"generation {generation - 1} best: {best['candidate_id']} "
                f"fitness={best['fitness']} selection_fitness={best['selection_fitness']}"
                f" violation={best['selection_violation']}"
                + (f" (feasible {feasible}/{len(scored)})" if is_constrained(ga_settings) else "")
            )
            # Claim the imports first: the offspring count is what is left over
            # once they have their slots, so nothing is bred only to be replaced.
            migrants = draw_migrants(args.migration_pool, algorithm, ga_settings, search_space)
            pending = ask_offspring(algorithm, int(algorithm.pop_size) - len(migrants))
            origins = [{"type": "offspring"}] * len(pending)
            reuses: list[dict[str, Any] | None] = [None] * len(pending)
            if migrants:
                pending = append_individuals(
                    pending, [entry["individual"] for entry, _ in migrants], search_space
                )
                reuse_enabled = reuse_migrant_baseline(ga_settings)
                if reuse_enabled and evaluation.get("evaluation_mode") != "differential":
                    raise SystemExit(
                        "ga_settings.reuse_migrant_baseline needs evaluation_mode=differential; "
                        f"{args.evaluation} reports {evaluation.get('evaluation_mode')!r}, which "
                        "runs no control run for a migrant to reuse"
                    )
                for entry, distance in migrants:
                    reuse = baseline_reuse_payload(entry, repetitions, reuse_enabled)
                    origins.append(with_baseline_reuse(migration_origin(entry, distance), reuse))
                    reuses.append(reuse)
                reused = sum(1 for reuse in reuses if reuse is not None)
                if reused:
                    print(f"migration: reusing the source island's baseline for {reused} contexts")
            save_state(ga_state_path, algorithm, pending, generation, snapshot, repetitions,
                       origins, reuses)
    else:
        generation = 0
        problem = TrafficContextProblem(search_space, is_constrained(ga_settings))
        algorithm = make_algorithm(ga_settings, constraints)
        algorithm.setup(problem, seed=base_seed, verbose=False)
        write_json(args.archive, {"entries": []})
        pending = algorithm.ask()
        origins = [{"type": "initial"}] * len(pending)
        # Generation 0 has no pool to import from, so nothing skips its control run.
        reuses = [None] * len(pending)
        save_state(ga_state_path, algorithm, pending, generation, snapshot, origins=origins)

    individuals = [decode_individual(entry.X, search_space) for entry in pending]

    candidate_scenes = [
        make_candidate_scene(false_scene, individual, index, generation, origins[index],
                             search_space, reuses[index])
        for index, individual in enumerate(individuals)
    ]

    write_json(
        args.out,
        {
            "generation": generation,
            # The GA's RNG descends continuously from base_seed at generation 0;
            # pymoo does not reseed per generation, so this reflects that single
            # origin seed rather than a value specific to this generation.
            "base_seed": base_seed,
            "candidate_scenes": candidate_scenes,
        },
    )
    print(f"wrote {args.out} (generation {generation}, {len(candidate_scenes)} candidates)")


if __name__ == "__main__":
    main()
