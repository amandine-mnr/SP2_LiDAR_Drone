import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

CSV_PATH = "metrics_log/drone_log_multitrack_2026-06-07_23h35m00s.csv"
SKIP_FIRST_SECONDS = 0.0 #for stabilization
VALID_THRESHOLD = 1.0 #lost threshold
PLOT_3D_TRAJECTORIES = False
PLOT_ERRORS = False

def detect_true_tracks(columns):
    suffixes = []
    for col in columns:
        match = re.fullmatch(r"true_x(?:\.(\d+))?", col)
        if match:
            number = match.group(1)
            suffix = "" if number is None else f".{number}"
            if f"true_y{suffix}" in columns and f"true_z{suffix}" in columns:
                suffixes.append(suffix)
    return sorted(suffixes, key=lambda s: 0 if s == "" else int(s[1:]) + 1)

def true_cols(suffix):
    return [f"true_x{suffix}", f"true_y{suffix}", f"true_z{suffix}"]

def estimate_delay(time, true_pos, est_pos):
    if len(time) < 3:
        return 0, np.nan, np.array([np.nan, np.nan])

    dt = np.nanmean(np.diff(time))
    best_lags = []

    for dim in [0, 1]:  #x and y
        true_signal = true_pos[:, dim] - np.nanmean(true_pos[:, dim])
        est_signal = est_pos[:, dim] - np.nanmean(est_pos[:, dim])

        if np.nanstd(true_signal) == 0 or np.nanstd(est_signal) == 0:
            best_lags.append(0)
            continue

        corr = correlate(est_signal, true_signal, mode="full")
        lags = np.arange(-len(true_signal) + 1, len(true_signal))
        best_lags.append(lags[np.nanargmax(corr)])

    best_lag_mean = int(np.round(np.mean(best_lags)))
    delay_sec = best_lag_mean * dt
    per_axis_delay = np.array(best_lags) * dt

    return best_lag_mean, delay_sec, per_axis_delay

def compute_true_derivatives(time, true_pos):
    true_vel = np.zeros_like(true_pos)
    true_acc = np.zeros_like(true_pos)

    for i in range(1, len(true_pos) - 1):
        dt = time[i + 1] - time[i - 1]
        if dt > 0:
            true_vel[i] = (true_pos[i + 1] - true_pos[i - 1]) / dt

        dt1 = time[i] - time[i - 1]
        dt2 = time[i + 1] - time[i]
        if dt1 > 0 and dt2 > 0:
            v_prev = (true_pos[i] - true_pos[i - 1]) / dt1
            v_next = (true_pos[i + 1] - true_pos[i]) / dt2
            true_acc[i] = (v_next - v_prev) / ((dt1 + dt2) / 2)

    return true_vel, true_acc


def assign_estimated_tracks_to_true_tracks(df, true_suffixes):
    assignments = {}
    distance_table = []

    for track_id, g in df.groupby("track_id"):
        est = g[["est_x", "est_y", "est_z"]].to_numpy(float)

        mean_distances = []
        for true_index, suffix in enumerate(true_suffixes):
            true = g[true_cols(suffix)].to_numpy(float)
            dist = np.linalg.norm(est - true, axis=1)
            mean_distances.append(np.nanmean(dist))

        assigned_true = int(np.nanargmin(mean_distances))
        assignments[track_id] = assigned_true

        row = {
            "track_id": track_id,
            "assigned_true_track": assigned_true,
            "assigned_true_columns": ",".join(true_cols(true_suffixes[assigned_true])),
            "mean_distance_to_assigned_true_m": mean_distances[assigned_true],
        }
        for true_index, d in enumerate(mean_distances):
            row[f"mean_distance_to_true_{true_index}_m"] = d
        distance_table.append(row)

    return assignments, pd.DataFrame(distance_table).sort_values(["assigned_true_track", "track_id"])

def compute_metrics_for_estimated_track(g, true_suffix):
    time = g["time"].to_numpy(float)
    true_pos = g[true_cols(true_suffix)].to_numpy(float)
    est_pos = g[["est_x", "est_y", "est_z"]].to_numpy(float)

    errors = np.linalg.norm(true_pos - est_pos, axis=1)
    valid_mask = errors <= VALID_THRESHOLD
    lost_mask = ~valid_mask

    valid_errors = errors[valid_mask]
    rmse = np.sqrt(np.mean(valid_errors ** 2)) if len(valid_errors) else np.nan
    mean_err = np.mean(valid_errors) if len(valid_errors) else np.nan
    max_err = np.max(valid_errors) if len(valid_errors) else np.nan

    axis_error = np.abs(true_pos - est_pos)
    mean_axis_error = (
        np.mean(axis_error[valid_mask], axis=0)
        if np.any(valid_mask)
        else np.array([np.nan, np.nan, np.nan])
    )

    dt_array = np.diff(time, prepend=time[0])
    lost_frames = int(np.sum(lost_mask))
    lost_time = np.sum(dt_array[lost_mask])

    true_vel, true_acc = compute_true_derivatives(time, true_pos)
    est_vel = g[["vel_x", "vel_y", "vel_z"]].to_numpy(float) if {"vel_x", "vel_y", "vel_z"}.issubset(g.columns) else np.full_like(true_vel, np.nan)
    est_acc = g[["acc_x", "acc_y", "acc_z"]].to_numpy(float) if {"acc_x", "acc_y", "acc_z"}.issubset(g.columns) else np.full_like(true_acc, np.nan)

    vel_error = np.linalg.norm(true_vel - est_vel, axis=1)
    acc_error = np.linalg.norm(true_acc - est_acc, axis=1)

    vel_rmse = np.sqrt(np.nanmean(vel_error[valid_mask] ** 2)) if np.any(valid_mask) else np.nan
    acc_rmse = np.sqrt(np.nanmean(acc_error[valid_mask] ** 2)) if np.any(valid_mask) else np.nan

    _, delay_sec, per_axis_delay = estimate_delay(time, true_pos, est_pos)

    return {
        "n_samples": len(g),
        "rmse_m": rmse,
        "mean_error_m": mean_err,
        "max_error_m": max_err,
        "mean_abs_x_error_m": mean_axis_error[0],
        "mean_abs_y_error_m": mean_axis_error[1],
        "mean_abs_z_error_m": mean_axis_error[2],
        "lost_frames": lost_frames,
        "lost_time_s": lost_time,
        "velocity_rmse_m_s": vel_rmse,
        "acceleration_rmse_m_s2": acc_rmse,
        "delay_s": delay_sec,
        "delay_x_s": per_axis_delay[0],
        "delay_y_s": per_axis_delay[1],
    }

def plot_xyz_per_true_track(df, true_suffixes, assignments):

    for true_index, suffix in enumerate(true_suffixes):
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        fig.suptitle(f"True pose track {true_index}: XYZ position and assigned estimated tracks")

        true_by_time = (
            df[["time"] + true_cols(suffix)]
            .drop_duplicates(subset=["time"])
            .sort_values("time")
        )

        labels = ["X [m]", "Y [m]", "Z [m]"]
        true_col_names = true_cols(suffix)

        for dim, ax in enumerate(axes):
            ax.plot(
                true_by_time["time"],
                true_by_time[true_col_names[dim]],
                label=f"True {labels[dim][0]}",
                linewidth=2,
            )

            assigned_track_ids = [
                tid for tid, assigned_true_index in assignments.items()
                if assigned_true_index == true_index
            ]

            for track_id in assigned_track_ids:
                g = df[df["track_id"] == track_id].sort_values("time")
                ax.plot(
                    g["time"],
                    g[["est_x", "est_y", "est_z"][dim]],
                    "--",
                    linewidth=1.4,
                    label=f"Est track {track_id}",
                )

            ax.set_ylabel(labels[dim])
            ax.grid(True)
            ax.legend(loc="best")

        axes[-1].set_xlabel("Time [s]")
        plt.tight_layout()


def plot_3d_per_true_track(df, true_suffixes, assignments):
    for true_index, suffix in enumerate(true_suffixes):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"3D trajectory - true pose track {true_index}")

        true_by_time = (
            df[["time"] + true_cols(suffix)]
            .drop_duplicates(subset=["time"])
            .sort_values("time")
        )

        ax.plot(
            true_by_time[f"true_x{suffix}"],
            true_by_time[f"true_y{suffix}"],
            true_by_time[f"true_z{suffix}"],
            label=f"True {true_index}",
            linewidth=2,
        )

        assigned_track_ids = [
            tid for tid, assigned_true_index in assignments.items()
            if assigned_true_index == true_index
        ]

        for track_id in assigned_track_ids:
            g = df[df["track_id"] == track_id].sort_values("time")
            ax.plot(
                g["est_x"],
                g["est_y"],
                g["est_z"],
                "--",
                label=f"Est track {track_id}",
            )

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

def plot_error_per_estimated_track(df, assignments, true_suffixes):
    plt.figure(figsize=(11, 5))
    for track_id, g in df.groupby("track_id"):
        g = g.sort_values("time")
        suffix = true_suffixes[assignments[track_id]]
        true_pos = g[true_cols(suffix)].to_numpy(float)
        est_pos = g[["est_x", "est_y", "est_z"]].to_numpy(float)
        errors = np.linalg.norm(true_pos - est_pos, axis=1)
        plt.plot(g["time"], errors, label=f"Est track {track_id} → true {assignments[track_id]}")

    plt.axhline(VALID_THRESHOLD, linestyle=":", label=f"Lost threshold = {VALID_THRESHOLD:.2f} m")
    plt.title("Position error of each estimated track against assigned true track")
    plt.xlabel("Time [s]")
    plt.ylabel("3D error [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def plot_whole_3D(df, true_suffixes, assignments):

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot all true trajectories
    true_idx0 = 0
    true_idx1 = 1
    true_idx2 = 2
    suffix0 = true_suffixes[0]
    suffix1 = true_suffixes[1]
    suffix2 = true_suffixes[2]
    true_df = (
        df[["time"] + true_cols(suffix0)]
        .drop_duplicates(subset=["time"])
        .sort_values("time")
    )

    ax.plot(
        true_df[f"true_x{suffix0}"],
        true_df[f"true_y{suffix0}"],
        true_df[f"true_z{suffix0}"],
        linewidth=3,
        label=f"True drone {true_idx0}"
    )

    true_df = (
        df[["time"] + true_cols(suffix1)]
        .drop_duplicates(subset=["time"])
        .sort_values("time")
    )

    ax.plot(
        true_df[f"true_x{suffix1}"],
        true_df[f"true_y{suffix1}"],
        true_df[f"true_z{suffix1}"],
        linewidth=2,
        label=f"True drone {true_idx1}"
    )

    true_df = (
        df[["time"] + true_cols(suffix2)]
        .drop_duplicates(subset=["time"])
        .sort_values("time")
    )

    ax.plot(
        true_df[f"true_x{suffix2}"],
        true_df[f"true_y{suffix2}"],
        true_df[f"true_z{suffix2}"],
        linewidth=1,
        label=f"True drone {true_idx2}"
    )

    # Plot all estimated tracks
    for track_id, g in df.groupby("track_id"):

        assigned_true = assignments[track_id]

        ax.plot(
            g["est_x"],
            g["est_y"],
            g["est_z"],
            "--",
            linewidth=1.5,
            label=f"Track {track_id} → True {assigned_true}"
        )

    ax.set_title("All True and Estimated Tracks")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.grid(True)
    ax.legend(loc="best")

    plt.tight_layout()

def main():
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(["time", "track_id"]).reset_index(drop=True)

    if SKIP_FIRST_SECONDS > 0:
        first_time = df["time"].min()
        df = df[df["time"] >= first_time + SKIP_FIRST_SECONDS].copy()

    true_suffixes = detect_true_tracks(df.columns)
    if not true_suffixes:
        raise ValueError("No true pose columns found")

    print(f"Detected {len(true_suffixes)} true pose tracks:")
    for i, suffix in enumerate(true_suffixes):
        print(f"  true {i}: {true_cols(suffix)}")

    assignments, assignment_table = assign_estimated_tracks_to_true_tracks(df, true_suffixes)

    print("\nEstimated track assignment:")
    print(assignment_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    metrics_rows = []
    for track_id, g in df.groupby("track_id"):
        assigned_true_index = assignments[track_id]
        suffix = true_suffixes[assigned_true_index]
        metrics = compute_metrics_for_estimated_track(g.sort_values("time"), suffix)
        metrics_rows.append({
            "track_id": track_id,
            "assigned_true_track": assigned_true_index,
            **metrics,
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["assigned_true_track", "track_id"])

    print("\nMetrics per estimated track:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Aggregate summary by true track, weighted by samples where useful
    summary_rows = []
    for true_index in range(len(true_suffixes)):
        sub = metrics_df[metrics_df["assigned_true_track"] == true_index]
        if len(sub) == 0:
            summary_rows.append({
                "true_track": true_index,
                "n_estimated_tracks_assigned": 0,
                "assigned_track_ids": "",
                "mean_rmse_m": np.nan,
                "mean_error_m": np.nan,
                "total_lost_frames": 0,
                "total_lost_time_s": 0.0,
            })
        else:
            summary_rows.append({
                "true_track": true_index,
                "n_estimated_tracks_assigned": len(sub),
                "assigned_track_ids": ",".join(map(str, sub["track_id"].tolist())),
                "mean_rmse_m": sub["rmse_m"].mean(),
                "mean_error_m": sub["mean_error_m"].mean(),
                "total_lost_frames": int(sub["lost_frames"].sum()),
                "total_lost_time_s": sub["lost_time_s"].sum(),
            })

    summary_df = pd.DataFrame(summary_rows)
    print("\nSummary by true pose track:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    assignment_table.to_csv("track_assignment.csv", index=False)
    metrics_df.to_csv("multidrone_metrics_per_estimated_track.csv", index=False)
    summary_df.to_csv("multidrone_metrics_by_true_track.csv", index=False)

    plot_xyz_per_true_track(df, true_suffixes, assignments)

    plot_whole_3D(df, true_suffixes, assignments)

    if PLOT_3D_TRAJECTORIES:
        plot_3d_per_true_track(df, true_suffixes, assignments)

    if PLOT_ERRORS:
        plot_error_per_estimated_track(df, assignments, true_suffixes)

    plt.show()


if __name__ == "__main__":
    main()
