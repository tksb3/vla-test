from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "input"
OUTPUT_DIR = REPO_ROOT / "output"

# One scenario's whole input set lives under input/scenarios/<name>/. Nothing is
# shared between scenarios: each directory carries its own common.json, so editing
# one scenario's SimLingo/CARLA/stall-guard settings cannot disturb another.
SCENARIOS_DIR = INPUT_DIR / "scenarios"
COMMON_CONFIG_NAME = "common.json"
SCENARIO_CONFIG_NAME = "config.json"
SCENARIO_INSTRUCTION_SPACE_NAME = "instruction_space.json"
SCENARIO_TRAFFIC_CONTEXT_SPACE_NAME = "traffic_context_space.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def available_scenarios() -> list[str]:
    """The scenario directories shipped with the repo, at any depth.

    Reported as repo-root-relative paths (input/scenarios/<name>/), because
    --scenario takes a path: listing a bare name would offer something that only
    resolves when the working directory happens to be input/scenarios/.
    """
    if not SCENARIOS_DIR.is_dir():
        return []
    return sorted(
        str(path.parent.relative_to(REPO_ROOT))
        for path in SCENARIOS_DIR.rglob(SCENARIO_CONFIG_NAME)
        if path.is_file()
    )


def scenario_input_paths(scenario: str) -> dict[str, Path]:
    """Resolve one scenario's input files from the directory holding them.

    `scenario` is a path -- absolute, or relative to the working directory -- to
    the directory carrying common.json, config.json, instruction_space.json and
    traffic_context_space.json. It is never interpreted as a name looked up
    somewhere else, so the same string means the same directory whether it comes
    from a shell, a script, or a scheduler.
    """
    scenario_dir = Path(scenario)
    if not (scenario_dir / SCENARIO_CONFIG_NAME).is_file():
        known = ", ".join(available_scenarios()) or "(none)"
        raise ValueError(
            f"no {SCENARIO_CONFIG_NAME} in scenario directory {scenario_dir}; "
            f"--scenario takes a path to the directory holding the input set. "
            f"Scenarios in this repo (relative to {REPO_ROOT}): {known}"
        )
    return {
        "config": scenario_dir / SCENARIO_CONFIG_NAME,
        "instruction_space": scenario_dir / SCENARIO_INSTRUCTION_SPACE_NAME,
        "traffic_context_space": scenario_dir / SCENARIO_TRAFFIC_CONTEXT_SPACE_NAME,
    }


def read_config(path: Path) -> dict[str, Any]:
    """Read a config, filling in the common.json sitting next to it.

    config.json carries what identifies the scenario; common.json next to it carries
    the SimLingo/CARLA/stall-guard blocks. config.json wins on any key it names.
    Both files belong to one scenario directory, so neither is shared with another
    scenario.
    """
    config = read_json(path)
    common_path = path.parent / COMMON_CONFIG_NAME
    if common_path.is_file() and path.resolve() != common_path.resolve():
        merged = read_json(common_path)
        merged.update(config)
        return merged
    return config


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_").lower() or "case"


def parse_route_xml(path: Path) -> dict[str, Any]:
    tree = ET.parse(path)
    root = tree.getroot()
    route = root.find("route")
    if route is None:
        raise ValueError(f"route XML has no <route>: {path}")

    waypoints = []
    waypoints_root = route.find("waypoints")
    if waypoints_root is not None:
        for position in waypoints_root.findall("position"):
            waypoints.append(
                {
                    "x": float(position.attrib.get("x", "0")),
                    "y": float(position.attrib.get("y", "0")),
                    "z": float(position.attrib.get("z", "0")),
                }
            )

    scenarios = []
    scenarios_root = route.find("scenarios")
    if scenarios_root is not None:
        for scenario in scenarios_root.findall("scenario"):
            entry = {
                "name": scenario.attrib.get("name"),
                "type": scenario.attrib.get("type"),
            }
            trigger_point = scenario.find("trigger_point")
            if trigger_point is not None:
                entry["trigger_point"] = {
                    "x": float(trigger_point.attrib.get("x", "0")),
                    "y": float(trigger_point.attrib.get("y", "0")),
                    "z": float(trigger_point.attrib.get("z", "0")),
                    "yaw": float(trigger_point.attrib.get("yaw", "0")),
                }
            scenarios.append(entry)

    return {
        "route_id": route.attrib.get("id"),
        "road_id": route.attrib.get("road_id"),
        "town": route.attrib.get("town"),
        "waypoints": waypoints,
        "scenarios": scenarios,
    }


def _format_xml_float(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _weather_attributes(weather: dict[str, Any]) -> dict[str, str]:
    return {key: _format_xml_float(float(value)) for key, value in weather.items()}


def _require_scenario_parameters(runtime_parameters: dict[str, Any]) -> dict[str, Any]:
    if "scenario_parameters" not in runtime_parameters:
        raise ValueError("scenario_runtime_parameters.scenario_parameters is required")
    return runtime_parameters["scenario_parameters"]


def _find_scenarios(route: ET.Element, traffic_scenario: str) -> list[ET.Element]:
    scenarios_root = route.find("scenarios")
    if scenarios_root is None:
        raise ValueError("route XML has no <scenarios>")
    scenarios = [
        scenario
        for scenario in scenarios_root.findall("scenario")
        if scenario.attrib.get("type") == traffic_scenario
    ]
    if not scenarios:
        raise ValueError(f"route XML has no scenario type={traffic_scenario}")
    return scenarios


def _set_value_child(scenario: ET.Element, name: str, value: float) -> None:
    """Write <name value="..."/>, the scenario_runner get_value_parameter shape."""
    child = scenario.find(name)
    if child is None:
        child = ET.SubElement(scenario, name)
    child.set("value", _format_xml_float(float(value)))


def _set_interval_child(scenario: ET.Element, name: str, interval: dict[str, Any]) -> None:
    """Write <name from="..." to="..."/>, the get_interval_parameter shape."""
    if "from" not in interval or "to" not in interval:
        raise ValueError(f"{name} requires from and to")
    child = scenario.find(name)
    if child is None:
        child = ET.SubElement(scenario, name)
    child.set("from", _format_xml_float(float(interval["from"])))
    child.set("to", _format_xml_float(float(interval["to"])))


def _require_keys(scenario_parameters: dict[str, Any], traffic_scenario: str, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in scenario_parameters]
    if missing:
        raise ValueError(
            f"{traffic_scenario} requires scenario_parameters.{', scenario_parameters.'.join(missing)}"
        )


def _apply_actor_flow_parameters(route: ET.Element, traffic_scenario: str, runtime_parameters: dict[str, Any]) -> None:
    """Shared by the scenarios whose knobs are flow_speed + source_dist_interval.

    Only those two elements are touched, so route-specific children such as
    EnterActorFlow's start_actor_flow / end_actor_flow survive untouched.
    """
    scenario_parameters = _require_scenario_parameters(runtime_parameters)
    _require_keys(scenario_parameters, traffic_scenario, ("flow_speed", "source_dist_interval"))

    for scenario in _find_scenarios(route, traffic_scenario):
        _set_value_child(scenario, "flow_speed", scenario_parameters["flow_speed"])
        _set_interval_child(scenario, "source_dist_interval", scenario_parameters["source_dist_interval"])


def _apply_signalized_junction_left_turn_parameters(route: ET.Element, runtime_parameters: dict[str, Any]) -> None:
    _apply_actor_flow_parameters(route, "SignalizedJunctionLeftTurn", runtime_parameters)


def _apply_non_signalized_junction_left_turn_parameters(
    route: ET.Element, runtime_parameters: dict[str, Any]
) -> None:
    _apply_actor_flow_parameters(route, "NonSignalizedJunctionLeftTurn", runtime_parameters)


def _apply_enter_actor_flow_parameters(route: ET.Element, runtime_parameters: dict[str, Any]) -> None:
    _apply_actor_flow_parameters(route, "EnterActorFlow", runtime_parameters)


def _apply_obstacle_parameters(route: ET.Element, traffic_scenario: str, runtime_parameters: dict[str, Any]) -> None:
    """Shared by the scenarios whose knobs are distance + speed."""
    scenario_parameters = _require_scenario_parameters(runtime_parameters)
    _require_keys(scenario_parameters, traffic_scenario, ("distance", "speed"))

    for scenario in _find_scenarios(route, traffic_scenario):
        _set_value_child(scenario, "distance", scenario_parameters["distance"])
        # SetMaxSpeed caps the background traffic (km/h), not the ego.
        _set_value_child(scenario, "speed", scenario_parameters["speed"])


def _apply_parked_obstacle_parameters(route: ET.Element, runtime_parameters: dict[str, Any]) -> None:
    _apply_obstacle_parameters(route, "ParkedObstacle", runtime_parameters)


def _apply_construction_obstacle_parameters(route: ET.Element, runtime_parameters: dict[str, Any]) -> None:
    _apply_obstacle_parameters(route, "ConstructionObstacle", runtime_parameters)


def _apply_parked_obstacle_two_ways_parameters(route: ET.Element, runtime_parameters: dict[str, Any]) -> None:
    scenario_parameters = _require_scenario_parameters(runtime_parameters)
    _require_keys(scenario_parameters, "ParkedObstacleTwoWays", ("distance", "frequency"))

    for scenario in _find_scenarios(route, "ParkedObstacleTwoWays"):
        _set_value_child(scenario, "distance", scenario_parameters["distance"])
        _set_interval_child(scenario, "frequency", scenario_parameters["frequency"])
        # ParkedObstacle's own knobs; the TwoWays subclass inherits them but the
        # Bench2Drive XML leaves them at the scenario_runner defaults.
        if "speed" in scenario_parameters:
            _set_value_child(scenario, "speed", scenario_parameters["speed"])


# traffic_scenario -> function writing that scenario's knobs into the route XML.
# Adding a scenario means adding one adapter above plus one line here.
ROUTE_PARAMETER_ADAPTERS = {
    "SignalizedJunctionLeftTurn": _apply_signalized_junction_left_turn_parameters,
    "NonSignalizedJunctionLeftTurn": _apply_non_signalized_junction_left_turn_parameters,
    "EnterActorFlow": _apply_enter_actor_flow_parameters,
    "ParkedObstacle": _apply_parked_obstacle_parameters,
    "ConstructionObstacle": _apply_construction_obstacle_parameters,
    "ParkedObstacleTwoWays": _apply_parked_obstacle_two_ways_parameters,
}


def _apply_traffic_scenario_parameters(
    route: ET.Element,
    traffic_scenario: str,
    runtime_parameters: dict[str, Any],
) -> None:
    adapter = ROUTE_PARAMETER_ADAPTERS.get(traffic_scenario)
    if adapter is None:
        raise ValueError(
            f"unsupported traffic_scenario for Step 03 route adapter: {traffic_scenario} "
            f"(registered: {', '.join(sorted(ROUTE_PARAMETER_ADAPTERS))})"
        )
    adapter(route, runtime_parameters)


def write_route_with_runtime_parameters(
    source_path: Path,
    output_path: Path,
    traffic_scenario: str,
    runtime_parameters: dict[str, Any],
) -> Path:
    tree = ET.parse(source_path)
    route = tree.getroot().find("route")
    if route is None:
        raise ValueError(f"route XML has no <route>: {source_path}")

    old_weathers = route.find("weathers")
    if old_weathers is not None:
        route.remove(old_weathers)

    if "weather" not in runtime_parameters:
        raise ValueError("scenario_runtime_parameters.weather is required")
    attrs = _weather_attributes(runtime_parameters["weather"])
    weathers = ET.SubElement(route, "weathers")
    start_attrs = dict(attrs)
    start_attrs["route_percentage"] = "0"
    ET.SubElement(weathers, "weather", start_attrs)
    end_attrs = dict(attrs)
    end_attrs["route_percentage"] = "100"
    ET.SubElement(weathers, "weather", end_attrs)

    _apply_traffic_scenario_parameters(route, traffic_scenario, runtime_parameters)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path.resolve()


def write_runtime_prompt(
    path: Path,
    instruction: str,
    tag: str | None = "instruction_following",
    enabled: bool = True,
) -> Path:
    payload = {
        "enabled": enabled,
        "prompt": instruction,
        "mode": "append",
        "tag": tag,
    }
    # Written atomically: the SimLingo agent re-reads this file every tick and
    # falls back to disabled on a partial read.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)
    return path


# Leaderboard infraction keys. min_speed_infractions is excluded: the criterion
# logs the ego/traffic speed ratio at every route checkpoint regardless of value,
# and Bench2Drive marks it 'unused' for scoring.
COLLISION_INFRACTION_KEYS = {"collisions_layout", "collisions_pedestrian", "collisions_vehicle"}
RED_LIGHT_INFRACTION_KEYS = {"red_light"}
ROUTE_DEVIATION_INFRACTION_KEYS = {"route_dev"}
IGNORED_INFRACTION_KEYS = {"min_speed_infractions"}


def _infraction_counts(infractions: Any) -> dict[str, float]:
    if isinstance(infractions, dict):
        counts = {}
        for key, value in infractions.items():
            if isinstance(value, list):
                counts[key] = float(len(value))
            elif isinstance(value, (int, float)):
                counts[key] = float(value)
            elif value:
                counts[key] = 1.0
        return {key: value for key, value in counts.items() if value}
    if isinstance(infractions, list):
        counts = {}
        for item in infractions:
            key = str(item)
            counts[key] = counts.get(key, 0.0) + 1.0
        return counts
    return {}


def evaluate_leaderboard_result(result: dict[str, Any]) -> dict[str, Any]:
    # SimLingo/Leaderboard result JSONs nest the actual record under _checkpoint.
    global_record = (result.get("_checkpoint") or {}).get("global_record") or {}

    status = result.get("status") or global_record.get("status")
    entry_status = result.get("entry_status")
    score_composed = result.get("score_composed")
    if score_composed is None:
        score_composed = result.get("scores", {}).get("score_composed")
    if score_composed is None:
        score_composed = (global_record.get("scores_mean") or {}).get("score_composed")

    infractions = result.get("infractions")
    if infractions is None:
        infractions = global_record.get("infractions")
    counts = _infraction_counts(infractions)
    scored_counts = {
        key: value for key, value in counts.items() if key not in IGNORED_INFRACTION_KEYS
    }

    has_collision = bool(COLLISION_INFRACTION_KEYS & scored_counts.keys())
    has_red_light = bool(RED_LIGHT_INFRACTION_KEYS & scored_counts.keys())
    has_route_deviation = bool(ROUTE_DEVIATION_INFRACTION_KEYS & scored_counts.keys())
    completed = status == "Completed" or entry_status == "Finished"

    # Collision outweighs the whole first term, which is capped at 100: within a
    # single run, any collision therefore sorts above any outcome without one.
    # Averaging the repetitions breaks that ordering again -- one collision in
    # three lands near 47, below a candidate that stalled all three times -- so
    # the ordering holds per run, not per candidate.
    #
    # Red-light and route-deviation terms used to sit here at +30 and +20. They
    # were dropped: a red light never fired once across 2400 candidates of
    # ParkedObstacle and NonSignalizedJunctionLeftTurn (the latter has no signal
    # by definition), and carrying a weight that only ever adds zero meant having
    # to justify a number Bench2Drive does not have. Route deviation fired on 14
    # of those candidates and is left to show up through score_composed, which it
    # already lowers by ending the route. A scenario where either matters needs
    # its weight chosen then, against data. Both flags stay in the returned
    # evaluation: they are still observations worth recording, just not scored.
    fitness = 0.0
    if isinstance(score_composed, (int, float)):
        fitness += max(0.0, 100.0 - float(score_composed))
    if has_collision:
        fitness += 100.0

    return {
        "status": status,
        "entry_status": entry_status,
        "score_composed": score_composed,
        "has_collision": has_collision,
        "has_red_light_violation": has_red_light,
        "has_route_deviation": has_route_deviation,
        "completed": completed,
        "infraction_count": len(scored_counts),
        "fitness": round(fitness, 3),
    }
