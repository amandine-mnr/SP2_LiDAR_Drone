import yaml
import itertools
import os
import signal
import subprocess
import time
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent
PARAM_TEMPLATE = PROJECT_ROOT / "ros2_ws" / "src" / "config" / "param.yaml"
SDF_TEMPLATE = PROJECT_ROOT / "crazyflie-simulation" / "simulator_files" / "gazebo" / "lidar_3D_template.sdf"
ACTIVE_SDF = PROJECT_ROOT / "crazyflie-simulation" / "simulator_files" / "gazebo" / "lidar_3D.sdf"
CONFIG_DIR = PROJECT_ROOT / "grid_configs"
RESULTS_FILE = PROJECT_ROOT / "grid_search_results.csv"
LOG_DIR = PROJECT_ROOT / "ros2_ws" / "src" / "lidar" / "lidar" / "metrics_log" 

LAUNCH_CMD = ["ros2", "launch", "start", "launch_grid.py"]

LOC_GRID = {
    "bg_threshold": [0.15],
    "dbscan_min_samples": [1],
    "memory_len": [7],
    "lidar_noise_std": [0.02],
    "size_margin": [0.30],
    "roi_factor": [0.7],
    "roi_max_radius": [2.5],
    "mahalanobis_gate": [11.34],
    "n_background_frames": [4],
}

TRAJ_GRID = {
    "base_altitude": [1.0],
    "speed_limit": [1.0],

    "max_length": [1.5],
    "max_width": [1.0],
    "max_delta_z": [0.1],

    "waypoint_spacing": [0.1],
    # "wp_threshold": [0.2],
    "kp": [1.5, 2.0],
}

LIDAR_GRID = {
    "horizontal_samples": [512, 1024, 2048],
    "vertical_samples": [64, 128, 256],
    "max_range": [75.0],
    "update_rate": [5, 20],
}

DRONE_START_POS = np.array([0.0, -4.0, 0.0])
LIDAR_POS = np.array([0.5, 1.5, 0.5])

MAX_RUNS = None
RUN_TIMEOUT_SEC = 300
COOLDOWN_SEC = 1
MIN_VALID_ROWS = 20



def load_base_config():
    with open(PARAM_TEMPLATE, "r") as f:
        return yaml.safe_load(f)


def write_config(base_config, loc_params, traj_params, run_id):
    CONFIG_DIR.mkdir(exist_ok=True)
    cfg = deepcopy(base_config)

    loc_ros_params = cfg["drone_localizer"]["ros__parameters"]
    for key, value in loc_params.items():
        loc_ros_params[key] = value

    traj_ros_params = cfg["crazyflie_polynomial_trajectory"]["ros__parameters"]
    for k, v in traj_params.items():
        traj_ros_params[k] = v

    out = CONFIG_DIR / f"config_{run_id:04d}.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out

def write_lidar_sdf(lidar_params):
    tree = ET.parse(SDF_TEMPLATE)
    root = tree.getroot()

    # horizontal samples
    h_samples = root.find(
        ".//lidar/scan/horizontal/samples"
    )
    h_samples.text = str(
        lidar_params["horizontal_samples"]
    )

    # vertical samples
    v_samples = root.find(
        ".//lidar/scan/vertical/samples"
    )
    v_samples.text = str(
        lidar_params["vertical_samples"]
    )

    # max range
    max_range = root.find(
        ".//lidar/range/max"
    )
    max_range.text = str(
        lidar_params["max_range"]
    )

    # update rate
    update_rate = root.find(
        ".//sensor/update_rate"
    )
    update_rate.text = str(
        lidar_params["update_rate"]
    )

    tree.write(ACTIVE_SDF)

def latest_log():
    if not LOG_DIR.exists():
        return None
    logs = list(LOG_DIR.glob("drone_log*.csv"))
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)

def compute_metrics(log_file):
    try:
        df = pd.read_csv(log_file)
    except Exception:
        return {"rmse": np.inf, "mean_error": np.inf, "max_error": np.inf, "valid_rows": 0, "lost_ratio": 1.0, "drone_true_speed" : np.inf, "mean_dist_to_lidar" : np.inf, "drone_pos": DRONE_START_POS[1]}

    required = ["true_x", "true_y", "true_z", "est_x", "est_y", "est_z"]
    if any(c not in df.columns for c in required):
        return {"rmse": np.inf, "mean_error": np.inf, "max_error": np.inf, "valid_rows": 0, "lost_ratio": 1.0, "drone_true_speed" : np.inf, "mean_dist_to_lidar" : np.inf, "drone_pos": DRONE_START_POS[1]}

    df = df.dropna(subset=required)
    if len(df) < MIN_VALID_ROWS:
        return {"rmse": np.inf, "mean_error": np.inf, "max_error": np.inf, "valid_rows": len(df), "lost_ratio": 1.0, "drone_true_speed" : np.inf, "mean_dist_to_lidar" : np.inf, "drone_pos": DRONE_START_POS[1]}

    # Position error
    err = np.sqrt(
        (df["true_x"] - df["est_x"]) ** 2
        + (df["true_y"] - df["est_y"]) ** 2
        + (df["true_z"] - df["est_z"]) ** 2
    )
    if "lost" in df.columns:
        valid_mask = (df["lost"].to_numpy() == 0)
    else:
        valid_mask = np.ones(len(df), dtype=bool)
    err = err[valid_mask]

    # Ground-truth speed
    mean_speed = np.nan
    # max_speed = np.nan
    if "time" in df.columns and len(df) > 2:
        dt = np.diff(df["time"].to_numpy())
        dx = np.diff(df["true_x"].to_numpy())
        dy = np.diff(df["true_y"].to_numpy())
        dz = np.diff(df["true_z"].to_numpy())
        valid = dt > 0
        speed = np.sqrt(dx**2 + dy**2 + dz**2) / dt
        speed = speed[valid]
        if len(speed):
            mean_speed = float(np.mean(speed))
            # max_speed = float(np.max(speed))

    # Mean distance to LiDAR
    dist_to_lidar = np.sqrt(
        (df["true_x"] - LIDAR_POS[0]) ** 2
        + (df["true_y"] - LIDAR_POS[1]) ** 2
        + (df["true_z"] - LIDAR_POS[2]) ** 2
    )
    mean_dist_to_lidar = float(np.mean(dist_to_lidar))

    # Lost
    if "lost" in df.columns:
        lost_ratio = float((df["lost"] > 0).mean())
    else:
        lost_ratio = np.nan

    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean_error": float(np.mean(err)),
        "max_error": float(np.max(err)),
        "valid_rows": int(len(df)),
        "lost_ratio": lost_ratio,
        "drone_true_speed": mean_speed,
        "mean_dist_to_lidar" : mean_dist_to_lidar,
        "drone_pos": DRONE_START_POS[1]
    }

def stop_process_group(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        time.sleep(3)
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass

def run_one(config_path):
    before = latest_log()

    cmd = LAUNCH_CMD + [f"config_file:={config_path}"]
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    try:
        proc.wait(timeout=RUN_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        stop_process_group(proc)
        print("timeout process")

    time.sleep(COOLDOWN_SEC)

    after = latest_log()
    if after is None or after == before:
        return None, {"rmse": np.inf, "mean_error": np.inf, "max_error": np.inf, "valid_rows": 0, "lost_ratio": 1.0, "drone_true_speed" : np.inf, "mean_dist_to_lidar" : np.inf, "drone_pos": DRONE_START_POS[1]}

    return str(after), compute_metrics(after)


def main():
    base_config = load_base_config()

    loc_keys = list(LOC_GRID.keys())
    lidar_keys = list(LIDAR_GRID.keys())
    traj_keys = list(TRAJ_GRID.keys())

    loc_combos = list(itertools.product(*LOC_GRID.values()))
    lidar_combos = list(itertools.product(*LIDAR_GRID.values()))
    traj_combos = list(itertools.product(*TRAJ_GRID.values()))

    all_combos = list(itertools.product(loc_combos, traj_combos, lidar_combos))

    if MAX_RUNS is not None:
        all_combos = all_combos[:MAX_RUNS]

    results = []

    for run_id, (loc_combo, traj_combo, lidar_combo) in enumerate(all_combos):
        loc_params = dict(zip(loc_keys, loc_combo))
        traj_params = dict(zip(traj_keys, traj_combo))
        lidar_params = dict(zip(lidar_keys, lidar_combo))

        print(f"\n=== Run {run_id + 1}/{len(all_combos)} ===")

        print("Loc params:")
        print(loc_params)

        print("Traj params:")
        print(traj_params)

        print("LiDAR params:")
        print(lidar_params)

        cfg_path = write_config(
            base_config,
            loc_params,
            traj_params,
            run_id,
        )

        write_lidar_sdf(lidar_params)

        log_file, metrics = run_one(
            cfg_path,
        )

        row = {
            "run_id": run_id,
            "config_file": str(cfg_path),
            "log_file": log_file,
            **metrics,
            **loc_params,
            **traj_params,
            **lidar_params,
        }
        results.append(row)

        df = pd.DataFrame(results).sort_values("rmse")
        df.to_csv(RESULTS_FILE, index=False)

        print(f"RMSE: {metrics['rmse']:.4f} | rows: {metrics['valid_rows']} | lost_ratio: {metrics['lost_ratio']:.3f} | drone_true_speed: {metrics['drone_true_speed']:.3f} | mean_dist_to_lidar: {metrics['mean_dist_to_lidar']:.3f} | drone_pos: {metrics['drone_pos']}")
        print("Current best:")

        print(
            df.head(3)[
                [
                    "rmse",
                    "mean_error",
                    "valid_rows",
                    *loc_keys,
                    *traj_keys,
                    *lidar_keys,
                ]
            ]
        )

    print(f"\nSaved results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
