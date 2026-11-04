#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/simlingo}
CARLA_ROOT=${CARLA_ROOT:-/opt/carla0915}
CHECKPOINT=${CHECKPOINT:-/workspace/checkpoints/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt}
ROUTE=${ROUTE:-${REPO_ROOT}/leaderboard/data/one_route_town10hd.xml}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/eval_results/one_route_town10hd}
HOST=${HOST:-localhost}
PORT=${PORT:-2000}
TM_PORT=${TM_PORT:-8000}
SEED=${SEED:-1}
SIMLINGO_RUNTIME_PROMPT=${SIMLINGO_RUNTIME_PROMPT:-${REPO_ROOT}/config/runtime_prompt.json}

mkdir -p "${OUT_ROOT}/viz" "${OUT_ROOT}/res" "${OUT_ROOT}/logs"

ROUTE_BASENAME="$(basename "${ROUTE}")"
ROUTE_COPY="${OUT_ROOT}/logs/${ROUTE_BASENAME}"
ROUTE_SHA256="$(sha256sum "${ROUTE}" | awk '{print $1}')"
CHECKPOINT_SHA256="missing"
if [[ -f "${CHECKPOINT}" ]]; then
  CHECKPOINT_SHA256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
fi

cp "${ROUTE}" "${ROUTE_COPY}"
cat > "${OUT_ROOT}/logs/run_metadata.txt" <<EOF
timestamp_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
repo_root=${REPO_ROOT}
carla_root=${CARLA_ROOT}
conda_env=${CONDA_ENV:-simlingo510}
route=${ROUTE}
route_copy=${ROUTE_COPY}
route_sha256=${ROUTE_SHA256}
checkpoint=${CHECKPOINT}
checkpoint_sha256=${CHECKPOINT_SHA256}
out_root=${OUT_ROOT}
host=${HOST}
port=${PORT}
tm_port=${TM_PORT}
seed=${SEED}
use_existing_carla=${USE_EXISTING_CARLA:-1}
simlingo_debug_viz=${SIMLINGO_DEBUG_VIZ:-1}
runtime_prompt=${SIMLINGO_RUNTIME_PROMPT}
EOF

source /workspace/miniconda/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-simlingo510}

export CARLA_ROOT
export WORK_DIR="${REPO_ROOT}"
export SCENARIO_RUNNER_ROOT="${REPO_ROOT}/Bench2Drive/scenario_runner"
export LEADERBOARD_ROOT="${REPO_ROOT}/Bench2Drive/leaderboard"
export PYTHONPATH="${REPO_ROOT}:${CARLA_ROOT}/PythonAPI/carla:${LEADERBOARD_ROOT}:${SCENARIO_RUNNER_ROOT}:${PYTHONPATH:-}"
export SAVE_PATH="${OUT_ROOT}/viz/"
export ROUTES="${ROUTE}"
export USE_EXISTING_CARLA=${USE_EXISTING_CARLA:-1}
export SIMLINGO_DEBUG_VIZ=${SIMLINGO_DEBUG_VIZ:-1}
export SIMLINGO_RUNTIME_PROMPT

python -u "${REPO_ROOT}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  --traffic-manager-port "${TM_PORT}" \
  --traffic-manager-seed "${SEED}" \
  --routes "${ROUTE}" \
  --repetitions 1 \
  --track SENSORS \
  --checkpoint "${OUT_ROOT}/res/one_route_res.json" \
  --debug-checkpoint "${OUT_ROOT}/res/one_route_live.txt" \
  --timeout 600 \
  --agent "${REPO_ROOT}/team_code/agent_simlingo.py" \
  --agent-config "${CHECKPOINT}" \
  2>&1 | tee "${OUT_ROOT}/logs/run.log"
