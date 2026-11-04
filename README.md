# Setup

1. Install SimLingo from https://github.com/renzka/simlingo.

2. Place the following files into the downloaded SimLingo directory, replacing
   the existing file where one is already there:

   - `simlingo/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py` (overwrite)
   - `simlingo/team_code/agent_simlingo.py` (overwrite)
   - `simlingo/tools/run_one_route_closed_loop.sh` (new file)

3. Set up the SimLingo environment and CARLA by following the instructions on the SimLingo
   GitHub page.

4. In `common.json` under the `input` folder, change the `run_script` and `host`
   keys to match your own environment.

5. Run the method with the commands below.

   For M=0 (difference objective / constrained objective):
   The command below is the example to run in the parked obstacle scenario with M=0 and difference objective.

   ```bash
   OUT=output/test_simlingo_diff_m0
   mkdir -p $OUT

   for i in 0 1 2 3 4 5; do
     nohup python -u src/run_pipeline.py \
       --scenario input/scenarios/parked_obstacle_2554_Town12_diff_m0 \
       --instruction-start $i --instructions 1 \
       --out-dir $OUT \
       --carla-port $((2000 + i*20)) --tm-port $((8000 + i*20)) \
       --run-simlingo > $OUT/independent_instr0$i.log 2>&1 &
     echo "instr_0$i PID $!"
   done
   ```

   For M=2/4 (difference objective / constrained objective):
   The command below is the example to run in the parked obstacle scenario with M=2 and difference objective.

   ```bash
   OUT=output/test_simlingo_diff_m2
   SHARED=output/test_simlingo_diff_m2/shared
   mkdir -p $OUT

   for i in 0 1 2 3 4 5; do
     nohup python -u src/run_pipeline.py \
       --scenario input/scenarios/parked_obstacle_2554_Town12_diff_m2 \
       --island-index $i --islands 6 \
       --shared-dir $SHARED \
       --start-at yyyy-mm-ddThh:mm:ssZ \
       --out-dir $OUT/island_0$i \
       --carla-port $((2000 + i*20)) --tm-port $((8000 + i*20)) \
       --run-simlingo > $OUT/island_0$i.log 2>&1 &
   done
   ```

# Input files

- Stored under the `input/scenarios` folder.
- Folder naming convention: `(scenario name)_(diff or const)_(m=0/2/4)`
- `diff` is the difference objective, `const` is the constrained objective.

# Source code

- Stored under `src/`. `run_pipeline.py` is the entry point: running that Python
  file executes the whole process end to end.
- `conda_packages.txt`, `environment.yml` and `pip_packages.txt` are stored
  under the `our_method` folder.
- The experiment environment versions are:

  ```
  Python 3.10.20
  Ubuntu 24.04.3 LTS
  CARLA 0.9.15
  ```
- Hardware spec (one VM is prepared for every three islands):

  ```
  Run in cloud virtual machine
  60 GB CPU memory, 16 vCPU (AMD Ryzen 9 9950X 16-Core Processor)
  RTX 5090
  ```


# Input and output correspondence

| Configuration | Scenario | Input Path | Output Path |
|---|---|---|---|
| diff/m=0 | parked_obstacle_2554_Town12 | input/scenarios/parked_obstacle_2554_Town12_diff_m0 | output/parked_obstacle_2554_Town12_diff_m0 |
| const/m=0 | parked_obstacle_2554_Town12 | input/scenarios/parked_obstacle_2554_Town12_const_m0 | output/parked_obstacle_2554_Town12_const_m0 |
| diff/m=2 | parked_obstacle_2554_Town12 | input/scenarios/parked_obstacle_2554_Town12_diff_m2 | output/parked_obstacle_2554_Town12_diff_m2 |
| const/m=2 | parked_obstacle_2554_Town12 | input/scenarios/parked_obstacle_2554_Town12_const_m2 | output/parked_obstacle_2554_Town12_const_m2 |
| diff/m=4 | parked_obstacle_2554_Town12 | input/scenarios/parked_obstacle_2554_Town12_diff_m4 | output/parked_obstacle_2554_Town12_diff_m4 |
| const/m=4 | parked_obstacle_2554_Town12 | input/scenarios/parked_obstacle_2554_Town12_const_m4 | output/parked_obstacle_2554_Town12_const_m4 |
| diff/m=0 | non_signalized_junction_left_turn_2084_Town12 | input/scenarios/non_signalized_junction_left_turn_2084_Town12_diff_m0 | output/non_signalized_junction_left_turn_2084_Town12_diff_m0 |
| const/m=0 | non_signalized_junction_left_turn_2084_Town12 | input/scenarios/non_signalized_junction_left_turn_2084_Town12_const_m0 | output/non_signalized_junction_left_turn_2084_Town12_const_m0 |
| diff/m=2 | non_signalized_junction_left_turn_2084_Town12 | input/scenarios/non_signalized_junction_left_turn_2084_Town12_diff_m2 | output/non_signalized_junction_left_turn_2084_Town12_diff_m2 |
| const/m=2 | non_signalized_junction_left_turn_2084_Town12 | input/scenarios/non_signalized_junction_left_turn_2084_Town12_const_m2 | output/non_signalized_junction_left_turn_2084_Town12_const_m2 |
| diff/m=4 | non_signalized_junction_left_turn_2084_Town12 | input/scenarios/non_signalized_junction_left_turn_2084_Town12_diff_m4 | output/non_signalized_junction_left_turn_2084_Town12_diff_m4 |
| const/m=4 | non_signalized_junction_left_turn_2084_Town12 | input/scenarios/non_signalized_junction_left_turn_2084_Town12_const_m4 | output/non_signalized_junction_left_turn_2084_Town12_const_m4 |

# Citations

SimLingo:
```
@InProceedings{Renz2025cvpr,
  title={SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment},
  author={Renz, Katrin and Chen, Long and Arani, Elahe and Sinavski, Oleg},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

Bench2Drive benchmark:
```
@inproceedings{Jia2024NeurIPS,
  title={Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving},
  author={Xiaosong Jia and Zhenjie Yang and Qifeng Li and Zhiyuan Zhang and Junchi Yan},
  booktitle={NeurIPS 2024 Datasets and Benchmarks Track},
  year={2024}
}
```