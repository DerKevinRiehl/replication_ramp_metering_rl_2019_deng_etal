# Replication study of Advanced Self-Improving Ramp Metering Algorithm based on Multi-Agent Deep Reinforcement Learning

## Authors

**Marco Prosperi**  
Institute for Transport Planning and Systems (IVT), ETH Zürich  
Replication project (Master Thesis)

## Introduction

This repository accompanies a replication of Deng et al. (2019), who propose a ramp metering controller trained with PPO/MAPPO from loop-detector occupancies. The original study reports that the RL agent outperforms no-control, fixed-time, and ALINEA baselines across three demand scenarios (stationary, flat peak, sharp peak) on single- and multi-ramp freeway networks simulated in SUMO.

**No code, data, or configuration files were released with the original paper.** We reconstructed the full pipeline from the published text, contacted the authors (no response), and calibrated undocumented simulation parameters through systematic grid searches. In ReScience C terms, this is an **unsuccessful quantitative replication ([¬Re])**: the methodological pipeline is recoverable and main qualitative trends are partially reproduced, but the exact numbers of Table I cannot be matched.

## The replicated study

```
Deng, F., Jin, J., Shen, Y., & Du, Y. (2019). Advanced self-improving ramp metering algorithm based on multi-agent deep reinforcement learning. In 2019 IEEE Intelligent Transportation Systems Conference (ITSC) (pp. 3804–3809). IEEE. https://doi.org/10.1109/ITSC.2019.8917353
```

## What this repository includes

```
./
├── 0_original_papers/              # Original ITSC paper (PDF + markdown)
├── 1_data_source/
│   ├── sumo_simulation_single_ramp/    # Single-ramp: stationary, flat_peak, sharp_peak
│   ├── sumo_simulation_multi_ramp/     # Two-ramp network (flat_peak)
│   └── sumo_simulation_single_ramp_alternatives/  # Sensitivity configs (IDM, EIDM, Krauss, τ, step)
├── 1_code_produce/
│   ├── config.py, sumo_env.py          # SUMO + Gymnasium interface
│   ├── train_ppo.py, train_mappo.py    # Single- and multi-ramp RL training
│   ├── evaluate_all.py                 # Table I evaluation
│   ├── evaluate_multi_ramp.py          # Multi-ramp ALINEA vs MAPPO
│   ├── enumerate_fixed_time.py, sweep_ocrit.py, sweep_depart_params.py
│   └── controllers/                    # no_control, fixed_time, alinea, ppo
├── 2_data_produce/
│   ├── models/                         # Trained PPO and MAPPO checkpoints
│   ├── logs/                           # TensorBoard + training checkpoints
│   └── results/                        # CSV results incl. comparison_with_paper.csv
├── 3_code_visualization/               # Scripts for demand, Table I, training, spacetime, multi-ramp figures
└── 3_data_visualization/               # Generated figures and spacetime caches
```

Pre-trained models and evaluation CSVs are included so results and figures can be regenerated without retraining.

## Installation Instructions

**Requirements:** Python 3.9+, SUMO v1.26.0, CPU only (no GPU needed).

```bash
# Install SUMO (example: Linux)
sudo apt-get install sumo sumo-tools sumo-doc
export SUMO_HOME="/usr/share/sumo"
export SUMO_BINARY="sumo"
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

# Verify
sumo --version

# Python dependencies
pip install stable-baselines3 gymnasium numpy pandas matplotlib tqdm tensorboard
```

On macOS, install SUMO via Homebrew and adjust `SUMO_HOME` accordingly. If TraCI imports fail, ensure `$SUMO_HOME/tools` is on `PYTHONPATH`. Optionally set `SUMO_BINARY` to override the default path in `config.py`.

## Run Instructions

Scripts print progress to the console; training also logs to TensorBoard.

```bash
# 1. Train single-ramp PPO on flat_peak (~6 h CPU, paper protocol)
cd 1_code_produce
python train_ppo.py --scenario flat_peak --episodes 1006
tensorboard --logdir ../2_data_produce/logs/tensorboard

# 2. Evaluate all controllers (10 episodes × 4 controllers × 3 scenarios)
python evaluate_all.py --scenarios all
# → 2_data_produce/results/comparison_with_paper.csv

# 3. Optional calibration (if rebuilding from scratch)
python enumerate_fixed_time.py --scenario all   # fixed-time grid search
python sweep_ocrit.py --scenario all            # ALINEA 396-run grid search

# 4. Multi-ramp MAPPO (~12 h CPU)
python train_mappo.py --timesteps 300000
python evaluate_multi_ramp.py

# 5. Figures
cd ../3_code_visualization
python plot_demand_profiles.py
python plot_table_comparison.py
python plot_training_paper.py
python plot_spacetime_grid_paper.py          # first run needs SUMO; caches .npz
python plot_spacetime_grid_paper.py --from-cache
python plot_multiramp_paper.py
```

## Replication Notes

### Outcome summary

| Aspect | Result |
|--------|--------|
| Pipeline reconstruction | Complete (SUMO v1.26.0 + TraCI + Stable-Baselines3 + Gymnasium) |
| Qualitative trends | Partially reproduced: PPO highest return in **2/3** scenarios; expected spacetime congestion patterns recovered |
| Table I numerics | **Not reproduced** — substantial speed/queue discrepancies remain |

**Our single-ramp results** (10-episode mean; full table in `comparison_with_paper.csv`):

| Scenario | Best controller (ours) | Return (ours / paper) |
|----------|------------------------|----------------------:|
| Stationary | **PPO** | 6,565 / 5,522 |
| Flat peak | **PPO** | 6,245 / 6,345 |
| Sharp peak | Fixed-Time (PPO close) | 6,900 / 6,964 (PPO: 6,739) |

PPO speeds exceed the paper by 16–25%; ramp queues saturate at ~145 m vs. 29–92 m in the original. No-control and fixed-time speeds fall *below* the paper, while PPO speeds fall *above* — indicating a different speed–queue trade-off frontier, not a uniform scaling error.

### What was easy

The RL formulation is clearly specified: state = K=4 profile occupancies (p2–p5); action = continuous metering rate (one-car-per-green, 3 s green); reward = v̄ − ηq̄ with η = 0.1 (Eq. 3).

### What was difficult

Reconstruction required substantial reverse-engineering because the paper omits:

- All vehicle type parameters (car-following model, τ, accel/decel, dimensions, departure mode)
- Numerical demand values (profiles reconstructed by visual interpolation from unlabelled figures; note also that in-text figure references are shifted by one from Fig. 4 onward)
- Network, route, and SUMO configuration files
- ALINEA parameters that transfer to our setup (396-run grid search needed despite paper's ō = 0.18, K_R = 0.35)
- Actor/Critic roles are swapped in Section II.C text (loss functions are correct)

### Key assumptions (after calibration)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Car-following | Krauss (SUMO default) | Preserves merge-dominated congestion; IDM attenuates metering effect |
| Time headway τ | 1.5 s | No-control speed ~38 km/h; metering controllers remain effective |
| Simulation step | 0.5 s | 120 steps per 60 s control period |
| Fixed-time rate | 515 veh/h (all scenarios) | Grid search minimising distance to Table I |
| ALINEA | ō = 0.14; K_R = 0.35 / 0.50 / 0.20 | Per-scenario grid search; downstream occupancy at p5 |
| PPO training | 1,006 episodes on flat_peak | Stable-Baselines3, MLP [64,64], lr = 3×10⁻² |
| MAPPO | 300,000 timesteps, flat_peak, 2 ramps | Shared policy, neighbour reward in state |

Sensitivity analysis across 8 alternative car-following configurations (`sumo_simulation_single_ramp_alternatives/`, `sweep_depart_params.py`) shows no-control speed ranging from 35 to 46 km/h on the same network — confirming that undocumented micro-parameters are first-order determinants of the reported performance.

### Original Table I (reference)

| Scenario | Controller | Speed | Queue | Return |
|----------|------------|------:|------:|-------:|
| Stationary | No-control | 38.62 | 4.57 | 4,579 |
| Stationary | Fixed-time | 47.84 | 39.49 | 5,266 |
| Stationary | ALINEA | 57.31 | 118.81 | 5,451 |
| Stationary | PPO | 55.24 | 92.26 | 5,522 |
| Flat peak | No-control | 43.08 | 4.45 | 5,116 |
| Flat peak | Fixed-time | 51.49 | 37.28 | 5,731 |
| Flat peak | ALINEA | 49.89 | 38.74 | 5,523 |
| Flat peak | PPO | 55.71 | 28.32 | 6,345 |
| Sharp peak | No-control | 54.66 | 1.47 | 6,541 |
| Sharp peak | Fixed-time | 59.76 | 38.37 | 6,711 |
| Sharp peak | ALINEA | 57.54 | 11.09 | 6,772 |
| Sharp peak | PPO | 60.97 | 29.34 | 6,964 |

## Citation

Replication Study:

```
[To be added after publication on ReScience C]
```

Original Paper:

```
Deng, F., Jin, J., Shen, Y., & Du, Y. (2019). Advanced self-improving ramp metering algorithm based on multi-agent deep reinforcement learning. In 2019 IEEE Intelligent Transportation Systems Conference (ITSC) (pp. 3804–3809). IEEE. https://doi.org/10.1109/ITSC.2019.8917353
```
