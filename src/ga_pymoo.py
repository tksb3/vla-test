"""pymoo-backed GA engine for step 02.

Step 02 runs once per generation as a separate process, so the algorithm state
has to survive between invocations. pymoo's ask/tell interface fits that: ask()
hands out the individuals to simulate, the CARLA runs happen in step 03/04, and
tell() feeds the fitness back on the next invocation. The algorithm object is
pickled in between.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA, FitnessSurvival, comp_by_cv_and_fitness
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.individual import Individual
from pymoo.core.mating import Mating
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection

# How many times the mating loop and the initial sampling may retry before
# giving up on producing feasible individuals.
MAX_FEASIBILITY_ITERATIONS = 100

# ga_settings.operator_parameters picks where the crossover/mutation rates come
# from: pymoo's own operator defaults, or the rates spelled out in ga_settings.
OPERATOR_PARAMETERS_PYMOO = "pymoo_defaults"
OPERATOR_PARAMETERS_GA_SETTINGS = "ga_settings"

# ga_settings keys that never reach pymoo, per mode. tournament_size cannot be
# honoured because the comparator only ever looks at two entrants, and
# elite_count has no equivalent because FitnessSurvival takes no arguments -- its
# elitism is structural (parents and offspring compete for pop_size slots).
# migration_target is left over from when migrants overwrote offspring already
# bred; the island now mates migration_count fewer of them, so there is no slot
# to choose.
UNUSED_GA_SETTINGS = {
    OPERATOR_PARAMETERS_PYMOO: (
        "mutation_rate",
        "crossover_rate",
        "tournament_size",
        "elite_count",
        "duplicate_retries",
        "migration_target",
    ),
    OPERATOR_PARAMETERS_GA_SETTINGS: (
        "tournament_size",
        "elite_count",
        "duplicate_retries",
        "migration_target",
    ),
}

# ga_settings.migration_selection decides which pooled contexts an island claims.
# 05_sync_island only pools contexts whose control run finished at fitness 0 on
# every repetition and that this island has not measured itself.
#
# 'random' and 'nearest' both draw from all of those and only differ in the
# ordering: nearest ranks by gene-space distance to the island's own surviving
# population, so imports land where this island is already working.
#
# 'positive_instruction' is not an ordering but a filter: it drops every pooled
# context whose instruction run also came out at fitness 0, then draws uniformly
# from what is left. Measured on the 0905 run, random imports carried an
# instruction mean of 3.9 against 76.5 for the island's own candidates -- almost
# every pooled context was one where neither run did anything, which is why
# migration came out net-negative there. The filter costs supply: of the pooled
# contexts, 57% had positive instruction fitness on NonSignalizedJunction 2084
# but only 1.7-13.5% on ParkedObstacle 2554, so on ParkedObstacle whole
# generations will import nothing.
MIGRATION_SELECTION_RANDOM = "random"
MIGRATION_SELECTION_NEAREST = "nearest"
MIGRATION_SELECTION_POSITIVE = "positive_instruction"
MIGRATION_SELECTIONS = (
    MIGRATION_SELECTION_RANDOM,
    MIGRATION_SELECTION_NEAREST,
    MIGRATION_SELECTION_POSITIVE,
)


def migration_selection_mode(settings: dict[str, Any]) -> str:
    """Which migrant-selection strategy ga_settings asks for, validated."""
    mode = settings.get("migration_selection", MIGRATION_SELECTION_RANDOM)
    if mode not in MIGRATION_SELECTIONS:
        raise SystemExit(
            f"ga_settings.migration_selection must be one of {list(MIGRATION_SELECTIONS)}, "
            f"got {mode!r}"
        )
    return mode


# ga_settings.selection_objective decides what the GA climbs.
#
# 'differential' keeps instruction-minus-baseline as a single objective, with no
# constraint: the control run's noise lands directly in the number being
# optimised, which doubles its variance (measured 992 -> 2451 on ParkedObstacle)
# and lets a candidate whose baseline happened to come out clean outrank one
# whose baseline happened not to.
#
# 'instruction_with_baseline_constraint' optimises the instruction run's own
# fitness and demands the control run never collide. A clean baseline becomes the
# price of admission rather than a term in the score, so baseline noise is
# reduced to a feasible/infeasible decision. pymoo ranks every feasible
# individual above every infeasible one (FitnessSurvival lexsorts on cv first),
# which is the intent -- a contradiction only reads as the instruction's doing
# when the control run drives clean.
SELECTION_DIFFERENTIAL = "differential"
SELECTION_INSTRUCTION_CONSTRAINED = "instruction_with_baseline_constraint"
SELECTION_OBJECTIVES = (SELECTION_DIFFERENTIAL, SELECTION_INSTRUCTION_CONSTRAINED)

# Violation recorded when the run produced no usable evaluation at all. Above the
# 1.0 a wholly-collided baseline reaches, so infrastructure failures sort behind
# every candidate whose genes were actually measured.
INFRASTRUCTURE_VIOLATION = 2.0


def selection_objective_mode(settings: dict[str, Any]) -> str:
    """Which objective/constraint split ga_settings asks for, validated."""
    mode = settings.get("selection_objective", SELECTION_DIFFERENTIAL)
    if mode not in SELECTION_OBJECTIVES:
        raise SystemExit(
            f"ga_settings.selection_objective must be one of {list(SELECTION_OBJECTIVES)}, "
            f"got {mode!r}"
        )
    return mode


def is_constrained(settings: dict[str, Any]) -> bool:
    """Whether the pymoo problem declares a constraint for this run.

    Frozen at generation 0: the problem is built once and pickled, so the number
    of constraints cannot change mid-run. verify_config_unchanged catches an
    edited setting before the next generation would disagree with the pickle.
    """
    return selection_objective_mode(settings) == SELECTION_INSTRUCTION_CONSTRAINED


def reuse_migrant_baseline(settings: dict[str, Any]) -> bool:
    """Whether an imported context may keep the baseline its source island measured.

    The control run never sees a prompt, so the same context measured on another
    island is measuring the same thing -- which is what makes skipping the rerun
    defensible. It lives in ga_settings rather than on the command line so the
    generation-0 snapshot freezes it: flipping it mid-run would put generations
    scored two different ways into the same tournament, and
    verify_config_unchanged stops that.
    """
    value = settings.get("reuse_migrant_baseline", False)
    if not isinstance(value, bool):
        raise SystemExit(
            f"ga_settings.reuse_migrant_baseline must be true or false, got {value!r}"
        )
    return value


def variable_names(search_space: dict[str, Any]) -> list[str]:
    """Array index -> gene name. The one ordering everything else agrees on.

    Sorted so it depends only on the set of genes, never on JSON key order:
    reformatting a search space must not silently permute the genotype a pickled
    population was built against.
    """
    for name, spec in search_space.items():
        if spec["type"] != "float":
            raise ValueError(
                f"unsupported GA variable type for gene {name}: {spec['type']}; "
                "every gene is a float, and an ordering between two of them goes in "
                "gene_constraints"
            )
    return sorted(search_space)


def variable_bounds(search_space: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """xl / xu in variable_names order."""
    names = variable_names(search_space)
    xl = np.array([float(search_space[name]["min"]) for name in names])
    xu = np.array([float(search_space[name]["max"]) for name in names])
    return xl, xu


def parse_constraints(space: dict[str, Any]) -> list[tuple[int, int]]:
    """gene_constraints as (low_index, high_index) pairs against the array layout.

    Each entry demands that one gene stay strictly below another -- the spawn
    interval's from below its to, say. Offspring that break it are thrown away
    and bred again rather than repaired, so the bounds of the two genes are free
    to overlap; the only thing that would make a constraint unsatisfiable is a
    high gene whose max does not exceed the low gene's min.
    """
    search_space = space["ga_search_space"]
    index = {name: position for position, name in enumerate(variable_names(search_space))}
    constraints = []
    for entry in space.get("gene_constraints", []):
        low, high = entry["low"], entry["high"]
        for name in (low, high):
            if name not in index:
                raise ValueError(f"gene_constraints names {name}, which is not in ga_search_space")
        if float(search_space[high]["max"]) <= float(search_space[low]["min"]):
            raise ValueError(
                f"gene_constraints requires {low} < {high}, but {high}.max "
                f"({search_space[high]['max']}) does not exceed {low}.min "
                f"({search_space[low]['min']}); no individual could satisfy it"
            )
        constraints.append((index[low], index[high]))
    return constraints


def is_feasible(x: np.ndarray, constraints: list[tuple[int, int]]) -> bool:
    """True when every ordered pair of genes is strictly increasing."""
    return all(x[high] > x[low] for low, high in constraints)


def encode_individual(individual: dict[str, Any], search_space: dict[str, Any]) -> np.ndarray:
    """Inverse of decode_individual: the pymoo genotype for an individual dict.

    Needed to drop a genotype that came from somewhere else -- an island's
    migration pool -- into the population. decode rounds to three decimals, so
    this does not round-trip exactly; the decoded values are what was simulated
    and therefore what the imported context means.
    """
    return np.array([float(individual[name]) for name in variable_names(search_space)])


def decode_individual(x: np.ndarray, search_space: dict[str, Any]) -> dict[str, Any]:
    """Turn a pymoo genotype into the individual dict the rest of the pipeline uses."""
    return {
        name: round(float(value), 3)
        for name, value in zip(variable_names(search_space), x)
    }


def normalize_genotypes(rows: np.ndarray, search_space: dict[str, Any]) -> np.ndarray:
    """Genotypes mapped onto the unit cube by each gene's own bounds.

    Raw distance would be dominated by whichever gene happens to span the widest
    range -- sun_azimuth_angle covers 360 while instruction_fire_distance_m
    covers 40 -- so every gene is put on the same footing before anything is
    measured. A gene whose bounds are a single point contributes nothing rather
    than dividing by zero.
    """
    xl, xu = variable_bounds(search_space)
    span = np.where(xu > xl, xu - xl, 1.0)
    return (np.atleast_2d(rows) - xl) / span


def distance_to_population(
    individual: dict[str, Any],
    population: Any,
    search_space: dict[str, Any],
) -> float:
    """Smallest normalized distance from one genotype to any member of a population.

    The minimum rather than a mean or a centroid: a population split across two
    regions has its centroid in a gap where nothing lives, and a context sitting
    right beside one of those clusters should read as near, not as halfway
    between both.
    """
    point = normalize_genotypes(encode_individual(individual, search_space), search_space)
    members = normalize_genotypes(
        np.array([member.X for member in population]), search_space
    )
    return float(np.min(np.linalg.norm(members - point, axis=1)))


class TrafficContextProblem(Problem):
    def __init__(self, search_space: dict[str, Any], constrained: bool = False) -> None:
        self.search_space = search_space
        xl, xu = variable_bounds(search_space)
        # n_ieq_constr has to be declared here for CV to mean anything: with none
        # declared, every individual's cv stays 0 and both the survival lexsort
        # and the tournament comparator fall through to the objective alone.
        super().__init__(n_var=len(xl), n_obj=1,
                         n_ieq_constr=1 if constrained else 0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):  # pragma: no cover - never called
        raise RuntimeError(
            "fitness comes from the CARLA runs in step 03/04; drive this problem via ask()/tell()"
        )


def binary_tournament(pop, P, random_state=None, **kwargs):
    """pymoo's own comparator, behind a name pickle can reference.

    comp_by_cv_and_fitness is built by @default_random_state, so its __qualname__
    contains <locals> and pickle cannot look it up -- which matters because the
    whole algorithm, this comparator included, is pickled between generations.
    A module-level delegate is referenced by name ('ga_pymoo',
    'binary_tournament'), and the closure itself never enters the pickle:
    importing this module rebuilds it on load.
    """
    return comp_by_cv_and_fitness(pop, P, random_state=random_state, **kwargs)


def operator_parameters_mode(settings: dict[str, Any]) -> str:
    mode = settings.get("operator_parameters", OPERATOR_PARAMETERS_PYMOO)
    if mode not in UNUSED_GA_SETTINGS:
        raise SystemExit(
            f"ga_settings.operator_parameters must be one of {sorted(UNUSED_GA_SETTINGS)}, got {mode!r}"
        )
    return mode


def unused_ga_settings(settings: dict[str, Any]) -> list[str]:
    return [key for key in UNUSED_GA_SETTINGS[operator_parameters_mode(settings)] if key in settings]


# ga_settings keys that only do something while the island exchange is running.
# A plain sweep is launched without --island-index, so run_pipeline passes step 02
# no --migration-pool and draw_migrants returns before reading any of them. They
# are easy to leave behind when a scenario directory is copied from an island run,
# where they read as if the sweep were importing contexts.
MIGRATION_ONLY_GA_SETTINGS = (
    "migration_count",
    "migration_selection",
    "reuse_migrant_baseline",
)


def inactive_migration_settings(settings: dict[str, Any], migrating: bool) -> list[str]:
    """Migration keys that are present but inert, because nothing is imported.

    Skips keys unused_ga_settings already reports, so a dead setting is named
    once rather than under two headings.
    """
    if migrating:
        return []
    reported = set(UNUSED_GA_SETTINGS[operator_parameters_mode(settings)])
    return [key for key in MIGRATION_ONLY_GA_SETTINGS
            if key in settings and key not in reported]


def build_operators(settings: dict[str, Any]) -> tuple[Any, Any]:
    """The crossover and mutation operators, from pymoo's defaults or ga_settings."""
    if operator_parameters_mode(settings) == OPERATOR_PARAMETERS_PYMOO:
        return SBX(), PM()

    crossover_rate = float(settings.get("crossover_rate", 0.9))
    mutation_rate = float(settings.get("mutation_rate", 0.2))
    # Crossover.prob is the chance the crossover runs at all; when it does not,
    # pymoo copies a parent through, matching the pre-pymoo crossover_rate.
    # prob=1.0 on the mutation leaves mutation_rate as the only gate, per gene.
    return SBX(prob=crossover_rate), PM(prob=1.0, prob_var=mutation_rate)


class FeasibleMating(Mating):
    """Mating that discards offspring violating a gene ordering.

    InfillCriterion.do() already loops until it has n_offsprings, re-running _do()
    for whatever is still missing, so dropping the infeasible ones here is all it
    takes to have the crossover redone. That loop never passes `parents` through,
    so every retry runs the tournament again and draws a different pair -- which
    matters, because one unlucky pairing could otherwise retry its way to the cap.

    Nothing is repaired. Swapping a from/to pair would evaluate a different
    traffic context than the crossover produced and would pile individuals up
    against the constraint boundary; rejecting keeps the population uniform over
    the feasible region.
    """

    def __init__(self, selection, crossover, mutation, constraints, **kwargs) -> None:
        super().__init__(selection, crossover, mutation, **kwargs)
        self.constraints = constraints

    def _do(self, problem, pop, n_offsprings, parents=None, random_state=None, **kwargs):
        off = super()._do(
            problem, pop, n_offsprings, parents=parents, random_state=random_state, **kwargs
        )
        if not self.constraints:
            return off
        keep = [index for index in range(len(off)) if is_feasible(off[index].X, self.constraints)]
        return off[keep]


class FeasibleSampling(FloatRandomSampling):
    """Uniform sampling restricted to the feasible region by rejection.

    Initialization.do() samples once and never retries, unlike the mating loop,
    so generation 0 has to reject and resample here or it would hand infeasible
    contexts straight to step 03.
    """

    def __init__(self, constraints, n_max_iterations: int = MAX_FEASIBILITY_ITERATIONS) -> None:
        super().__init__()
        self.constraints = constraints
        self.n_max_iterations = n_max_iterations

    def _do(self, problem, n_samples, random_state=None, **kwargs):
        if not self.constraints:
            return super()._do(problem, n_samples, random_state=random_state, **kwargs)

        rows: list[np.ndarray] = []
        for _ in range(self.n_max_iterations):
            batch = super()._do(problem, n_samples, random_state=random_state, **kwargs)
            rows.extend(row for row in batch if is_feasible(row, self.constraints))
            if len(rows) >= n_samples:
                return np.array(rows[:n_samples])
        raise SystemExit(
            f"could not sample {n_samples} feasible individuals in {self.n_max_iterations} "
            "attempts; check gene_constraints against the search space bounds"
        )


def ask_offspring(algorithm: Any, n_offsprings: int) -> Any:
    """ask() for exactly n_offsprings individuals.

    n_offsprings is a plain attribute pymoo reads in GeneticAlgorithm._infill, so
    lowering it for a generation that imports migrants leaves room for them
    without mating offspring only to overwrite them. It is set on every call
    because it survives in the pickle, and the count changes with how many
    contexts the pool can supply.
    """
    if n_offsprings < 1:
        raise SystemExit(
            f"n_offsprings must be >= 1, got {n_offsprings}; migration_count has to stay "
            "below population_size or a generation would be pure migration"
        )
    algorithm.n_offsprings = n_offsprings
    population = algorithm.ask()
    # The mating loop stops at n_max_iterations whether or not it filled the
    # order, and pymoo only warns when verbose. A short generation would leave
    # the origins list and the population out of step, so fail here instead.
    if len(population) < n_offsprings:
        raise SystemExit(
            f"mating produced {len(population)} of {n_offsprings} offspring before hitting "
            f"n_max_iterations={MAX_FEASIBILITY_ITERATIONS}; the gene_constraints are "
            "probably unsatisfiable over most of the search space -- check the bounds"
        )
    return population


def append_individuals(population: Any, individuals: list[dict[str, Any]],
                       search_space: dict[str, Any]) -> Any:
    """Extend an ask() result with genotypes that did not come from mating.

    The merged individuals carry no fitness yet, exactly like the offspring, so
    tell() feeds F back for all of them alike and pymoo never has to know some
    of them were imported rather than bred.
    """
    if not individuals:
        return population
    imported = Population.create(
        *[Individual(X=encode_individual(individual, search_space)) for individual in individuals]
    )
    return Population.merge(population, imported)


def make_algorithm(settings: dict[str, Any], constraints: list[tuple[int, int]]) -> GA:
    crossover, mutation = build_operators(settings)
    return GA(
        pop_size=int(settings.get("population_size", 8)),
        sampling=FeasibleSampling(constraints),
        survival=FitnessSurvival(),
        eliminate_duplicates=True,
        # Passing mating= makes GA ignore its own selection/crossover/mutation
        # arguments, so everything the operators need is configured here.
        mating=FeasibleMating(
            # The comparator only ever looks at P[i, 0] and P[i, 1], so a pressure
            # above 2 would silently drop the extra entrants.
            selection=TournamentSelection(func_comp=binary_tournament, pressure=2),
            crossover=crossover,
            mutation=mutation,
            constraints=constraints,
            eliminate_duplicates=DefaultDuplicateElimination(),
            n_max_iterations=MAX_FEASIBILITY_ITERATIONS,
        ),
    )


def make_config_snapshot(
    search_space: dict[str, Any],
    ga_settings: dict[str, Any],
    gene_constraints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """What generation 0 baked into the pymoo algorithm (frozen inside the pickle).

    The search space fixes the variable bounds and the array order genotypes are
    written in; gene_constraints fix which offspring get rejected; ga_settings
    shape the operators. All three are only read at generation 0
    (make_algorithm/algorithm.setup) -- editing for_02_..._space.json between
    generations does NOT reconfigure the pickled algorithm, so a later evolve
    call must detect drift instead of silently decoding new genes against old
    (or old genes against new) rules.
    """
    return {
        "search_space": search_space,
        "ga_settings": ga_settings,
        "gene_constraints": gene_constraints or [],
    }


def _flatten_diff(old: Any, new: Any, path: str, diffs: list[str]) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            _flatten_diff(old.get(key), new.get(key), f"{path}.{key}" if path else str(key), diffs)
    elif old != new:
        diffs.append(f"{path}: {old!r} -> {new!r}")


def describe_snapshot_diff(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    diffs: list[str] = []
    _flatten_diff(old, new, "", diffs)
    return diffs


def save_state(
    path: Path,
    algorithm: Any,
    pending: Any,
    generation: int,
    snapshot: dict[str, Any],
    repetitions: int | None = None,
    origins: list[dict[str, Any]] | None = None,
    baseline_reuses: list[dict[str, Any] | None] | None = None,
) -> None:
    """Persist the GA between the per-generation invocations of step 02.

    `origins` records where each pending individual came from -- mating, initial
    sampling, or another island's pool. It rides along because a run that dies
    after ask() re-emits this population without redrawing, and the pool entries
    that fed it are already marked used by then.

    `baseline_reuses` rides along for the same reason, and has to: the control
    run a migrant is to be scored against came out of a pool entry now marked
    used, so a re-emitted population could not look it up again.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "algorithm": algorithm,
                "pending": pending,
                "generation": generation,
                "snapshot": snapshot,
                "repetitions": repetitions,
                "origins": origins,
                "baseline_reuses": baseline_reuses,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            f"GA state {path} is missing; rerun step 02 without --evaluation to start a new search"
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


def verify_config_unchanged(
    state: dict[str, Any],
    search_space: dict[str, Any],
    ga_settings: dict[str, Any],
    gene_constraints: list[dict[str, Any]] | None = None,
) -> None:
    current = make_config_snapshot(search_space, ga_settings, gene_constraints)
    diffs = describe_snapshot_diff(state["snapshot"], current)
    if diffs:
        changes = "\n".join(f"  - {d}" for d in diffs)
        raise SystemExit(
            "traffic-context-space changed since generation 0, but the running GA "
            "(population_size, search bounds, operators) is frozen inside ga_state.pkl "
            "from generation 0 and does not pick up edits mid-run:\n"
            f"{changes}\n"
            "Revert the file to continue this run, or start a fresh search (new --out-dir "
            "or delete this instruction's ga_state.pkl/archive/population) to apply the change."
        )


def verify_repetitions_unchanged(state: dict[str, Any], repetitions: int) -> None:
    """Every generation must average over the same number of runs.

    `repetitions` lives in the scenario config, which this step never reads, so
    it is carried in from step 04's evaluation JSON and compared against what
    earlier generations used. Changing it mid-run would make generations with
    different noise levels compete on the same scale.
    """
    previous = state.get("repetitions")
    if previous is not None and int(previous) != int(repetitions):
        raise SystemExit(
            f"repetitions changed from {previous} to {repetitions} mid-run; earlier generations "
            "averaged over a different number of runs and are not comparable. Revert the config "
            "to continue this run, or start a fresh search to apply the change."
        )
