"""Evaluate multi-ramp flat_peak: ALINEA vs MAPPO with period-wise paper-style plots."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import traci
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CONTROL_STEPS,
    SIM_STEP,
    GREEN_PHASE,
    MIN_RED,
    MAX_RED,
    MR_MIN,
    MR_MAX,
    ETA,
    RANDOM_SEED,
    NUM_CONTROL_PERIODS,
    SUMO_BINARY,
)

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None


DATA_DIR = Path(__file__).parent.parent / "1_data_source" / "sumo_simulation_multi_ramp" / "flat_peak"
SUMO_CFG = str(DATA_DIR / "Configuration.sumocfg")
TLS_IDS = ["ramp_meter", "ramp_meter2"]
QUEUE_IDS = ["queue_ramp_1", "queue_ramp_2"]

OBS_DETECTORS = {
    0: {
        "p2": ["det_r1_p2_l0", "det_r1_p2_l1", "det_r1_p2_l2"],
        "p3": ["det_r1_p3_l0", "det_r1_p3_l1", "det_r1_p3_l2"],
        "p4": ["det_r1_p4_l1", "det_r1_p4_l2", "det_r1_p4_l3"],
        "p5": ["det_r1_p5_l0", "det_r1_p5_l1", "det_r1_p5_l2"],
    },
    1: {
        "p2": ["det_r2_p2_l0", "det_r2_p2_l1", "det_r2_p2_l2"],
        "p3": ["det_r2_p3_l0", "det_r2_p3_l1", "det_r2_p3_l2"],
        "p4": ["det_r2_p4_l1", "det_r2_p4_l2", "det_r2_p4_l3"],
        "p5": ["det_r2_p5_l0", "det_r2_p5_l1", "det_r2_p5_l2"],
    },
}

P4_MAINLINE = {
    0: ["det_r1_p4_l1", "det_r1_p4_l2", "det_r1_p4_l3"],
    1: ["det_r2_p4_l1", "det_r2_p4_l2", "det_r2_p4_l3"],
}


class AlineaLocal:
    def __init__(self, o_crit: float = 0.18, k_r: float = 0.35, initial_action: float = 0.5):
        self.o_crit = o_crit
        self.k_r = k_r
        self.initial_action = initial_action
        self.current_action = initial_action

    def reset(self):
        self.current_action = self.initial_action

    def act(self, o_down: float) -> float:
        delta = self.k_r * (self.o_crit - o_down)
        self.current_action = float(np.clip(self.current_action + delta, 0.0, 1.0))
        return self.current_action


class SharedMAPPO:
    def __init__(self, model_path: str):
        if PPO is None:
            raise ImportError("stable-baselines3 is required for PPO evaluation")
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"MAPPO model not found: {self.model_path}")
        self.model = PPO.load(str(self.model_path))

    def act_single(self, obs: np.ndarray) -> float:
        action, _ = self.model.predict(obs.reshape(1, -1), deterministic=True)
        action = np.asarray(action).flatten()
        return float(np.clip(action[0], 0.0, 1.0))


def _find_sumo_binary(use_gui: bool = True) -> str:
    if use_gui:
        sumo_dir = os.path.dirname(SUMO_BINARY)
        gui = os.path.join(sumo_dir, "sumo-gui")
        if os.path.exists(gui + ".exe"):
            return gui + ".exe"
        if os.path.exists(gui):
            return gui
        return "sumo-gui"
    return SUMO_BINARY


def _read_mean_occupancy(detector_ids: List[str]) -> float:
    values = []
    for det_id in detector_ids:
        try:
            occ = traci.inductionloop.getLastIntervalOccupancy(det_id)
            values.append(occ / 100.0)
        except traci.exceptions.TraCIException:
            pass
    return float(np.mean(values)) if values else 0.0


def _read_mean_speed_kmh(detector_ids: List[str]) -> float:
    values = []
    for det_id in detector_ids:
        try:
            speed_ms = traci.inductionloop.getLastIntervalMeanSpeed(det_id)
            if speed_ms >= 0:
                values.append(speed_ms * 3.6)
        except traci.exceptions.TraCIException:
            pass
    return float(np.mean(values)) if values else 0.0


def _read_queue_m(queue_id: str) -> float:
    try:
        return float(traci.lanearea.getJamLengthMeters(queue_id))
    except traci.exceptions.TraCIException:
        return 0.0


def _get_obs_ramp(ramp_idx: int) -> np.ndarray:
    d = OBS_DETECTORS[ramp_idx]
    return np.array([
        _read_mean_occupancy(d["p2"]),
        _read_mean_occupancy(d["p3"]),
        _read_mean_occupancy(d["p4"]),
        _read_mean_occupancy(d["p5"]),
    ], dtype=np.float32)


def _metering_rate_to_red_duration(metering_rate: float) -> float:
    if metering_rate <= 0:
        return MAX_RED
    cycle_time = 3600.0 / metering_rate
    red_duration = cycle_time - GREEN_PHASE
    return float(np.clip(red_duration, MIN_RED, MAX_RED))


def _apply_metering_rate(tls_id: str, action: float):
    metering_rate = MR_MIN + action * (MR_MAX - MR_MIN)
    red_duration = _metering_rate_to_red_duration(metering_rate)

    try:
        links = traci.trafficlight.getControlledLinks(tls_id)
        n_links = len(links)
    except traci.exceptions.TraCIException:
        n_links = 4

    if n_links < 1:
        return

    state_green = "G" * n_links
    state_red = "r" + "G" * (n_links - 1)

    logic = traci.trafficlight.Logic(
        programID="multi_eval",
        type=0,
        currentPhaseIndex=0,
        phases=[
            traci.trafficlight.Phase(duration=GREEN_PHASE, state=state_green),
            traci.trafficlight.Phase(duration=red_duration, state=state_red),
        ],
    )
    traci.trafficlight.setProgramLogic(tls_id, logic)


def run_episode(
    controller: str,
    model_path: str = None,
    seed: int = RANDOM_SEED,
    use_gui: bool = True,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    if controller == "mappo":
        if model_path is None:
            model_path = str(Path(__file__).parent.parent / "2_data_produce" / "models" / "mappo_multi_flat_peak.zip")
        policy = SharedMAPPO(model_path)
    elif controller == "alinea":
        agents = [AlineaLocal(), AlineaLocal()]
    else:
        raise ValueError(f"Unsupported controller for multi-ramp evaluation: {controller}")

    cmd = [
        _find_sumo_binary(use_gui),
        "-c", SUMO_CFG,
        "--start",
        "--quit-on-end",
        "--no-warnings",
        "--no-step-log",
        "--step-length", str(SIM_STEP),
        "--seed", str(seed),
    ]

    label = f"multi_ramp_{controller}_{seed}_{id(cmd)}"
    traci.start(cmd, label=label)
    traci.switch(label)

    reward_r1, reward_r2 = [], []
    speed_r1, speed_r2 = [], []
    queue_r1, queue_r2 = [], []

    last_r1 = 0.0
    last_r2 = 0.0

    for _ in range(CONTROL_STEPS):
        traci.simulationStep()

    done = False
    control_period = 0
    max_periods = NUM_CONTROL_PERIODS

    while not done:
        if controller == "mappo":
            # Normalize neighbor reward to [0,1] to match training env
            obs0 = np.concatenate([_get_obs_ramp(0), [np.clip(last_r2 / 100.0, 0.0, 1.0)]]).astype(np.float32)
            obs1 = np.concatenate([_get_obs_ramp(1), [np.clip(last_r1 / 100.0, 0.0, 1.0)]]).astype(np.float32)
            
            a0 = policy.act_single(obs0)
            a1 = policy.act_single(obs1)
        else:
            o_down_0 = _read_mean_occupancy(P4_MAINLINE[0])
            o_down_1 = _read_mean_occupancy(P4_MAINLINE[1])
            a0 = agents[0].act(o_down_0)
            a1 = agents[1].act(o_down_1)

        _apply_metering_rate(TLS_IDS[0], a0)
        _apply_metering_rate(TLS_IDS[1], a1)

        for _ in range(CONTROL_STEPS):
            traci.simulationStep()

        speed0 = _read_mean_speed_kmh(P4_MAINLINE[0])
        speed1 = _read_mean_speed_kmh(P4_MAINLINE[1])
        queue0 = _read_queue_m(QUEUE_IDS[0])
        queue1 = _read_queue_m(QUEUE_IDS[1])

        last_r1 = speed0 - ETA * queue0
        last_r2 = speed1 - ETA * queue1

        reward_r1.append(last_r1)
        reward_r2.append(last_r2)
        speed_r1.append(speed0)
        speed_r2.append(speed1)
        queue_r1.append(queue0)
        queue_r2.append(queue1)

        control_period += 1
        done = control_period >= max_periods

    traci.close()

    flat_speed = speed_r1 + speed_r2
    flat_queue = queue_r1 + queue_r2
    total_return = float(np.sum(np.array(reward_r1) + np.array(reward_r2))) if reward_r1 else 0.0

    stats = {
        "avg_speed_kmh": float(np.mean(flat_speed)) if flat_speed else 0.0,
        "avg_queue_m": float(np.mean(flat_queue)) if flat_queue else 0.0,
        "total_return": total_return,
    }
    traces = {
        "reward_r1": reward_r1,
        "reward_r2": reward_r2,
        "speed_r1": speed_r1,
        "speed_r2": speed_r2,
        "queue_r1": queue_r1,
        "queue_r2": queue_r2,
    }
    return stats, traces


def _smooth(values: List[float], window: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if window <= 1 or len(x) <= 2:
        return x
    window = min(window, len(x))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="same")


def plot_period_curves(alinea_traces: Dict[str, List[float]], ppo_traces: Dict[str, List[float]], out_path: Path, smooth_window: int = 5):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(alinea_traces["reward_r1"]))
    metrics = [
        ("reward", "Reward", "reward_r1", "reward_r2"),
        ("speed", "Speed (km/h)", "speed_r1", "speed_r2"),
        ("queue", "Queue (m)", "queue_r1", "queue_r2"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
    ppo_color = "#f2c94c"
    alinea_color = "#2d9cdb"

    for ax, (_, ylabel, key_r1, key_r2) in zip(axes, metrics):
        ar1 = _smooth(alinea_traces[key_r1], smooth_window)
        ar2 = _smooth(alinea_traces[key_r2], smooth_window)
        pr1 = _smooth(ppo_traces[key_r1], smooth_window)
        pr2 = _smooth(ppo_traces[key_r2], smooth_window)

        ax.plot(x, ar1, color=alinea_color, linestyle="-", label="ALINEA ramp1", linewidth=2)
        ax.plot(x, ar2, color=alinea_color, linestyle="--", label="ALINEA ramp2", linewidth=2)
        ax.plot(x, pr1, color=ppo_color, linestyle="-", label="MAPPO ramp1", linewidth=2)
        ax.plot(x, pr2, color=ppo_color, linestyle="--", label="MAPPO ramp2", linewidth=2)

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Current period")
        ax.set_xlim(0, min(125, len(x) - 1) if len(x) > 0 else 125)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate_multi(
    controllers: List[str],
    episodes: int,
    model_path: str = None,
    seed: int = RANDOM_SEED,
    use_gui: bool = True,
    plot_paper_curves: bool = False,
    smooth_window: int = 5,
):
    print("=" * 72)
    print("Multi-Ramp Evaluation (flat_peak): MAPPO vs ALINEA")
    print("=" * 72)
    print(f"Controllers: {controllers}")
    print(f"Episodes: {episodes}")
    print(f"Config: {SUMO_CFG}")

    rows = []
    first_episode_traces = {}
    for controller in controllers:
        ep_stats = []
        for ep in range(episodes):
            stats, traces = run_episode(
                controller=controller,
                model_path=model_path,
                seed=seed + ep,
                use_gui=use_gui,
            )
            ep_stats.append(stats)
            if ep == 0:
                first_episode_traces[controller] = traces

        speed = np.mean([x["avg_speed_kmh"] for x in ep_stats])
        queue = np.mean([x["avg_queue_m"] for x in ep_stats])
        total_return = np.mean([x["total_return"] for x in ep_stats])

        rows.append({
            "scenario": "multi_flat_peak",
            "controller": controller,
            "avg_speed_kmh": float(speed),
            "avg_queue_m": float(queue),
            "total_return": float(total_return),
            "episodes": episodes,
        })

        print(f"\n{controller}")
        print(f"  Speed  : {speed:.2f} km/h")
        print(f"  Queue  : {queue:.2f} m")
        print(f"  Return : {total_return:.2f}")

    import pandas as pd

    df = pd.DataFrame(rows)
    out_path = Path(__file__).parent.parent / "2_data_produce" / "results" / "multi_flat_peak_mappo_vs_alinea.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    if plot_paper_curves and "alinea" in first_episode_traces and "mappo" in first_episode_traces:
        fig_path = Path(__file__).parent.parent / "3_data_visualization" / "spacetime_diagrams" / "multi_flat_peak_period_curves.png"
        plot_period_curves(
            alinea_traces=first_episode_traces["alinea"],
            ppo_traces=first_episode_traces["mappo"],
            out_path=fig_path,
            smooth_window=smooth_window,
        )
        print(f"Paper-style period curves saved to: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-ramp flat_peak (MAPPO vs ALINEA)")
    parser.add_argument("--controllers", nargs="+", default=["alinea", "mappo"], choices=["alinea", "mappo"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--model-path", type=str, default=None, help="Multi-ramp MAPPO model path")
    parser.add_argument("--plot-paper-curves", action="store_true", help="Generate period-wise smoothed curves figure")
    parser.add_argument("--smooth-window", type=int, default=5, help="Moving-average window for plots")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui")
    args = parser.parse_args()

    evaluate_multi(
        controllers=args.controllers,
        episodes=args.episodes,
        model_path=args.model_path,
        seed=args.seed,
        use_gui=args.gui,
        plot_paper_curves=args.plot_paper_curves,
        smooth_window=max(1, args.smooth_window),
    )


if __name__ == "__main__":
    main()
