from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import OUTPUT_DIR, evaluate_leaderboard_result, read_json, write_json


TERMINATED_STATUSES = {"terminated_stall_guard", "terminated_timeout"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 4: evaluate simulation result.")
    parser.add_argument("--simulation", type=Path, default=OUTPUT_DIR / "03_carla_simulation" / "latest_simulation.json")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "04_evaluation.json")
    return parser.parse_args()


def termination_info(simulation_case: dict[str, Any]) -> dict[str, Any]:
    status = simulation_case.get("status")
    stall_guard = simulation_case.get("stall_guard") or {}
    forced = status in TERMINATED_STATUSES
    reason = None
    if status == "terminated_timeout":
        reason = "simulation_timeout"
    elif status == "terminated_stall_guard":
        reason = stall_guard.get("reason") or "ego_stationary"

    return {
        "forced_termination": forced,
        "termination_status": status if forced else None,
        "termination_reason": reason,
        "stall_guard_triggered": bool(stall_guard.get("triggered", False)),
        "stall_guard": stall_guard or None,
    }


def missing_result_evaluation(simulation_case: dict[str, Any]) -> dict[str, Any]:
    termination = termination_info(simulation_case)
    status = simulation_case.get("status")
    note = "No SimLingo result JSON was found."
    if termination["forced_termination"]:
        note = "No SimLingo result JSON was found because the simulation was forcibly terminated."
    elif status == "prepared":
        note = "No SimLingo result JSON was found because step 03 was run without --run-simlingo."

    return {
        "status": status or "not_run",
        "entry_status": None,
        "score_composed": None,
        "has_collision": False,
        "has_red_light_violation": False,
        "has_route_deviation": False,
        "completed": False,
        "infraction_count": 0,
        "fitness": 0.0,
        **termination,
        "note": note,
    }


def evaluate_simulation_case(simulation_case: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(simulation_case["result_path"])
    if result_path.exists():
        raw_result = read_json(result_path)
        evaluation = evaluate_leaderboard_result(raw_result)
        evaluation.update(termination_info(simulation_case))
        source = str(result_path)
    else:
        raw_result = None
        evaluation = missing_result_evaluation(simulation_case)
        source = None

    return {
        "case_id": simulation_case.get("case_id"),
        "candidate_index": simulation_case.get("candidate_index"),
        "simulation": simulation_case,
        "result_source": source,
        "evaluation": evaluation,
        "raw_result": raw_result,
    }


INFRASTRUCTURE_FAILURE_STATUSES = {"evaluation_failed", "terminated_carla_unreachable"}

COUNTED_FLAGS = (
    "has_collision",
    "has_red_light_violation",
    "has_route_deviation",
    "completed",
    "forced_termination",
)


def is_usable(evaluation: dict[str, Any]) -> bool:
    """A repetition counts towards the mean unless the infrastructure failed it.

    Stalls and crashes are real driving outcomes and stay in; only runs whose
    result could not be obtained at all are dropped.
    """
    return evaluation.get("status") not in INFRASTRUCTURE_FAILURE_STATUSES


def repetition_index(report: dict[str, Any]) -> int:
    """Which repetition this report came from. Runs predating the setting are 0."""
    return int((report.get("simulation") or {}).get("repetition", 0))


def aggregate_repetitions(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold the repetitions of one (case_id, role) into a single report.

    `fitness` becomes the mean over the usable repetitions; the per-run boolean
    outcomes cannot be averaged, so they are kept as counts alongside an OR-style
    flag. The individual fitness values are kept so the spread stays inspectable,
    and `fitness_by_repetition` keeps them addressable by repetition index so the
    differential can pair the two roles run by run.
    """
    usable = [report for report in reports if is_usable(report["evaluation"])]
    representative = dict(usable[0] if usable else reports[0])
    evaluations = [report["evaluation"] for report in usable]

    aggregated = dict(representative["evaluation"])
    aggregated["repetitions"] = len(reports)
    aggregated["usable_repetitions"] = len(usable)

    if evaluations:
        fitness_values = [float(evaluation.get("fitness") or 0.0) for evaluation in evaluations]
        aggregated["fitness"] = round(sum(fitness_values) / len(fitness_values), 3)
        aggregated["fitness_values"] = fitness_values
        # Keyed by str because this survives a JSON round-trip, where int keys would not.
        aggregated["fitness_by_repetition"] = {
            str(repetition_index(report)): value for report, value in zip(usable, fitness_values)
        }
        for flag in COUNTED_FLAGS:
            count = sum(1 for evaluation in evaluations if evaluation.get(flag))
            aggregated[f"{flag}_count"] = count
            aggregated[flag] = count > 0 if flag != "completed" else count == len(evaluations)
    else:
        # Every repetition hit an infrastructure failure; keep the failing status
        # so selection_fitness in step 02 still penalises this candidate.
        aggregated["fitness"] = 0.0
        aggregated["fitness_values"] = []
        aggregated["fitness_by_repetition"] = {}

    representative["evaluation"] = aggregated
    representative["repetition_reports"] = reports
    return representative


def merge_repetitions(case_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by (case_id, role) and collapse each group to one report."""
    grouped: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    order: list[tuple[str, str | None]] = []
    for report in case_reports:
        simulation = report.get("simulation") or {}
        key = (report["case_id"], simulation.get("role"))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(report)
    return [aggregate_repetitions(grouped[key]) for key in order]


def paired_differential(
    instruction_evaluation: dict[str, Any],
    baseline_evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Average the per-repetition instruction-minus-baseline differences.

    Pairing by repetition index rather than subtracting the two means matters as
    soon as the repetitions stop being interchangeable: once a run varies the
    seed per repetition, repetition k of both roles shares that seed, so the
    within-pair difference isolates the prompt's effect from the seed's. With a
    fixed seed the two agree whenever both roles have the same usable
    repetitions, and differ only when one role lost runs to infrastructure
    failures -- where pairing is the safer of the two.

    A repetition counts only when it is usable on both sides. If nothing pairs
    up, fall back to the difference of means so the candidate still gets a score
    rather than silently dropping out of selection.
    """
    instruction_by_repetition = instruction_evaluation.get("fitness_by_repetition") or {}
    baseline_by_repetition = baseline_evaluation.get("fitness_by_repetition") or {}
    shared = sorted(
        set(instruction_by_repetition) & set(baseline_by_repetition),
        key=int,
    )

    if not shared:
        difference_of_means = round(
            float(instruction_evaluation.get("fitness") or 0.0)
            - float(baseline_evaluation.get("fitness") or 0.0),
            3,
        )
        return {
            "fitness_differential": difference_of_means,
            "fitness_differential_values": [],
            "paired_repetitions": 0,
            "differential_method": "difference_of_means",
        }

    differences = [
        round(float(instruction_by_repetition[key]) - float(baseline_by_repetition[key]), 3)
        for key in shared
    ]
    return {
        "fitness_differential": round(sum(differences) / len(differences), 3),
        "fitness_differential_values": differences,
        "paired_repetitions": len(differences),
        "differential_method": "paired_repetitions",
    }


def reused_baseline_evaluation(payload: dict[str, Any]) -> dict[str, Any]:
    """The baseline block for a candidate whose control run was not rerun here.

    Step 03 leaves the baseline role unrun for an imported context and passes the
    source island's measurement through on the instruction case. Rebuilt in the
    shape aggregate_repetitions produces, so everything downstream -- the paired
    differential, the per-role outcome columns, step 02's infrastructure-failure
    check -- reads it the same way as a measured one. baseline_source marks it so
    05_sync_island refuses to republish it as this island's own.
    """
    evaluation = dict(payload.get("evaluation") or {})
    evaluation["baseline_source"] = "reused"
    evaluation["reused_from"] = {
        "source_island": payload.get("source_island"),
        "source_generation": payload.get("source_generation"),
        "source_candidate_id": payload.get("source_candidate_id"),
    }
    return evaluation


def merge_differential_pairs(case_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold instruction/baseline case pairs into one entry per candidate.

    The merged entry keeps the instruction run's shape (so downstream joins by
    case_id keep working) and embeds the baseline run plus the differential
    fitness inside `evaluation`.
    """
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    order: list[str] = []
    for report in case_reports:
        case_id = report["case_id"]
        if case_id not in by_case:
            by_case[case_id] = {}
            order.append(case_id)
        role = (report.get("simulation") or {}).get("role") or "instruction"
        by_case[case_id][role] = report

    merged = []
    for case_id in order:
        pair = by_case[case_id]
        instruction_report = pair.get("instruction")
        baseline_report = pair.get("baseline")
        if instruction_report is None:
            # No instruction run recorded; keep the baseline report so the case
            # stays visible, but it carries no differential fitness.
            merged.append(baseline_report)
            continue
        entry = dict(instruction_report)
        evaluation = dict(entry["evaluation"])
        reuse = (instruction_report.get("simulation") or {}).get("baseline_reuse")
        if baseline_report is not None:
            baseline_evaluation = baseline_report["evaluation"]
            evaluation["baseline"] = baseline_evaluation
            evaluation.update(paired_differential(evaluation, baseline_evaluation))
            entry["baseline_simulation"] = baseline_report["simulation"]
            entry["baseline_result_source"] = baseline_report.get("result_source")
            entry["baseline_raw_result"] = baseline_report.get("raw_result")
        elif reuse is not None:
            # No control run happened for this context: it was imported, and the
            # source island's measurement stands in for one.
            baseline_evaluation = reused_baseline_evaluation(reuse)
            evaluation["baseline"] = baseline_evaluation
            differential = paired_differential(evaluation, baseline_evaluation)
            differential["differential_method"] += "_reused_baseline"
            evaluation.update(differential)
        else:
            evaluation["baseline"] = None
        entry["evaluation"] = evaluation
        merged.append(entry)
    return merged


def main() -> None:
    args = parse_args()
    simulation = read_json(args.simulation)

    if "cases" in simulation:
        case_reports = [evaluate_simulation_case(case) for case in simulation["cases"]]
        # Repetitions collapse first, so the differential pairing below always
        # sees one instruction report and one baseline report per case.
        case_reports = merge_repetitions(case_reports)
        if any((case.get("role") for case in simulation["cases"])):
            case_reports = merge_differential_pairs(case_reports)
        report = {
            "step": "04_evaluate",
            "simulation": simulation,
            "evaluation_mode": simulation.get("evaluation_mode", "single"),
            "repetitions": simulation.get("repetitions", 1),
            # Lifted out of `simulation` so a per-generation tally of how many
            # CARLA runs a generation actually cost is one key away, without
            # walking the whole batch summary.
            "run_counts": simulation.get("run_counts"),
            "evaluations": case_reports,
        }
    else:
        case_report = evaluate_simulation_case(simulation)
        report = {
            "step": "04_evaluate",
            **case_report,
        }

    write_json(args.out, report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
