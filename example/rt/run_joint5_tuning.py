#!/usr/bin/env python3
"""Guarded joint-5 tuning runner for optimal_backstepping_torque.

Each real-robot trial requires an explicit operator confirmation. Trial configs,
console output, CSV logs and metrics are stored in a timestamped directory.
"""

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from plot_optimal_backstepping import OptimalBacksteppingLog


PARAM_BOUNDS = {
    # 目标位移：约 1.7～6.9 度
    "traj_amp5": (0.03, 0.20),

    # 运动频率：约 0.1～0.4 rad/s
    "traj_omega5": (0.10, 0.40),

    # 增大误差反馈，但限制最高值，避免高频振荡
    "k1_5": (4.0, 16.0),
    "k2_5": (6.0, 24.0),

    # 允许更明显的期望速度
    "alpha_limit5": (0.10, 0.30),

    # 保留滤波后的速度变化限制
    "alpha_dot_limit5": (0.30, 0.90),

    # 从当前 0.4 提高，但暂不直接放到很大
    "virtual_accel_limit5": (0.60, 1.80),

    # 继续抑制 alpha_dot 微分噪声
    "alpha_dot_filter": (0.005, 0.015),

    # 适度提高第五关节力矩上限
    "tau_limit5": (0.80, 2.00),

    # 避免命令形成过于明显的锯齿
    "tau_rate_limit5": (10.0, 24.0),

    # 鲁棒自适应项 nu_r=theta_hat*phi^2*z2/(2*iota^2)。
    "theta_initial": (0.02, 2.00),
    "theta_max": (0.50, 20.00),
    "gamma_theta": (0.02, 20.00),
    "sigma_theta": (0.001, 0.20),
    "learning_start_time": (1.00, 5.00),
}

def clamp(value, bounds):
    return max(bounds[0], min(bounds[1], value))


def replace_scalar(text, key, value):
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*).*$", re.MULTILINE)
    updated, count = pattern.subn(rf"\g<1>{value:.8g}", text)
    if count != 1:
        raise ValueError(f"配置项 {key} 应恰好出现一次，实际为 {count}")
    return updated


def replace_bool(text, key, value):
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*).*$", re.MULTILINE)
    updated, count = pattern.subn(
        rf"\g<1>{'true' if value else 'false'}", text)
    if count != 1:
        raise ValueError(f"配置项 {key} 应恰好出现一次，实际为 {count}")
    return updated


def replace_joint5(text, key, value):
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)([^#\n]+)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"找不到配置项 {key}")
    values = match.group(2).split()
    if len(values) != 6:
        raise ValueError(f"配置项 {key} 不是六轴数组")
    values = ["0", "0", "0", "0", f"{value:.8g}", "0"]
    return text[: match.start()] + match.group(1) + " ".join(values) + text[match.end() :]


def write_trial_config(base_text, params, path):
    text = base_text
    text = replace_joint5(text, "traj_amp", params["traj_amp5"])
    text = replace_joint5(text, "traj_omega", params["traj_omega5"])
    text = replace_joint5(text, "traj_phase", 0.0)
    vector_params = {
        "k1": "k1_5",
        "k2": "k2_5",
        "alpha_limit": "alpha_limit5",
        "alpha_dot_limit": "alpha_dot_limit5",
        "virtual_accel_limit": "virtual_accel_limit5",
        "tau_limit": "tau_limit5",
        "tau_rate_limit": "tau_rate_limit5",
    }
    for key, param_name in vector_params.items():
        text = replace_joint5(text, key, params[param_name])
    text = replace_scalar(text, "alpha_dot_filter", params["alpha_dot_filter"])
    text = replace_bool(text, "enable_theta_adaptation", True)
    for key in ("theta_initial", "theta_max", "gamma_theta", "sigma_theta",
                "learning_start_time"):
        text = replace_scalar(text, key, params[key])
    path.write_text(text, encoding="utf-8")


def values(rows, key):
    return [float(row[key]) for row in rows]


def analyze_log(path, params):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("日志没有数据行")

    error = values(rows, "error5")
    torque_raw = values(rows, "tau_raw5")
    torque_cmd = values(rows, "tau_cmd5")
    velocity = values(rows, "dq5")
    barrier = values(rows, "barrier_ratio5")
    alpha_dot = values(rows, "alpha_dot5")
    theta = values(rows, "theta_hat")
    torque_steps = [torque_cmd[i] - torque_cmd[i - 1] for i in range(1, len(rows))]
    flips = sum(torque_steps[i] * torque_steps[i - 1] < 0
                for i in range(1, len(torque_steps)))
    return {
        "samples": len(rows),
        "duration_s": float(rows[-1]["time"]),
        "aborted": any(float(row["abort"]) != 0.0 for row in rows),
        "error_rms": math.sqrt(sum(x * x for x in error) / len(error)),
        "error_peak": max(map(abs, error)),
        "velocity_peak": max(map(abs, velocity)),
        "tau_raw_peak": max(map(abs, torque_raw)),
        "tau_cmd_peak": max(map(abs, torque_cmd)),
        "tau_saturation_fraction": sum(
            abs(x) >= 0.98 * params["tau_limit5"] for x in torque_cmd) / len(rows),
        "torque_step_peak": max(map(abs, torque_steps), default=0.0),
        "torque_step_sign_flips": flips,
        "barrier_peak": max(map(abs, barrier)),
        "alpha_dot_peak": max(map(abs, alpha_dot)),
        "theta_initial_logged": theta[0],
        "theta_final": theta[-1],
        "theta_min": min(theta),
        "theta_peak": max(theta),
        "theta_change": theta[-1] - theta[0],
        "theta_max_fraction": sum(
            x >= 0.98 * params["theta_max"] for x in theta) / len(theta),
    }


def next_params(current, metrics):
    nxt = dict(current)
    oscillating = (metrics["torque_step_sign_flips"] > metrics["samples"] * 0.12 or
                   metrics["tau_saturation_fraction"] > 0.08)
    unsafe = (metrics["aborted"] or metrics["velocity_peak"] > 0.27 or
              metrics["barrier_peak"] > 0.82)

    if unsafe:
        nxt["traj_amp5"] *= 0.80
        nxt["traj_omega5"] *= 0.90
        nxt["k1_5"] *= 0.90
        nxt["k2_5"] *= 0.85
        nxt["virtual_accel_limit5"] *= 0.85
        nxt["theta_initial"] *= 0.80
        nxt["theta_max"] *= 0.85
        nxt["gamma_theta"] *= 0.70
        nxt["sigma_theta"] *= 1.25
        nxt["learning_start_time"] *= 1.15
    elif oscillating:
        nxt["k2_5"] *= 0.90
        nxt["alpha_dot_filter"] *= 0.80
        nxt["tau_rate_limit5"] *= 0.90
        nxt["gamma_theta"] *= 0.80
        nxt["sigma_theta"] *= 1.20
    elif metrics["error_rms"] > 0.01 and metrics["tau_saturation_fraction"] < 0.02:
        nxt["k1_5"] *= 1.10
        nxt["k2_5"] *= 1.05
        nxt["theta_initial"] *= 1.20
        nxt["theta_max"] *= 1.25
        nxt["gamma_theta"] *= 1.50
        nxt["sigma_theta"] *= 0.85
        nxt["learning_start_time"] *= 0.85
    else:
        nxt["traj_amp5"] *= 1.05

    for name, bounds in PARAM_BOUNDS.items():
        nxt[name] = clamp(nxt[name], bounds)
    return nxt


def initial_params():
    return {
        "traj_amp5": 0.10,
        "traj_omega5": 0.30,
        "k1_5": 8.0,
        "k2_5": 12.0,
        "alpha_limit5": 1.9,
        "alpha_dot_limit5": 0.90,
        "virtual_accel_limit5": 2.00,
        "alpha_dot_filter": 0.01,
        "tau_limit5": 1.00,
        "tau_rate_limit5": 16.0,
        "theta_initial": 0.20,
        "theta_max": 5.00,
        "gamma_theta": 0.50,
        "sigma_theta": 0.02,
        "learning_start_time": 2.00,
    }


def validate_initial_params(params):
    missing = set(PARAM_BOUNDS) - set(params)
    extra = set(params) - set(PARAM_BOUNDS)
    if missing or extra:
        raise ValueError(
            f"initial_params 与 PARAM_BOUNDS 键不一致: missing={sorted(missing)}, "
            f"extra={sorted(extra)}")
    for name, value in params.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"initial_params[{name}] 必须是有限数值")
        if value <= 0:
            raise ValueError(f"initial_params[{name}] 必须大于 0")
    if params["theta_initial"] > params["theta_max"]:
        raise ValueError("theta_initial 不能大于 theta_max")

def main():
    script_dir = Path(__file__).resolve().parent
    repo = script_dir.parent.parent
    parser = argparse.ArgumentParser(description="第五关节真机半自动调参")
    parser.add_argument("--trials", type=int, default=10, choices=range(1, 11))
    parser.add_argument("--config", type=Path,
                        default=script_dir / "optimal_backstepping_torque.cfg")
    parser.add_argument("--binary", type=Path,
                        default=repo / "build/bin/optimal_backstepping_torque")
    args = parser.parse_args()

    config = args.config.resolve()
    binary = args.binary.resolve()
    if not config.is_file() or not binary.is_file():
        print("找不到配置或控制器可执行文件，请先完成编译。", file=sys.stderr)
        return 2

    output_dir = script_dir / "tuning_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True)
    base_text = config.read_text(encoding="utf-8")
    params = initial_params()
    validate_initial_params(params)
    results = []

    def update_best():
        valid = [item for item in results
                 if not item["aborted"] and not item["console_abort"]]
        if not valid:
            return
        best = min(valid, key=lambda item: item["error_rms"])
        best_dir = output_dir / f"trial_{best['trial']:02d}"
        shutil.copy2(best_dir / "optimal_backstepping_torque.cfg",
                     output_dir / "best.cfg")
        try:
            best_log = OptimalBacksteppingLog(best_dir / "optimal_backstepping_log.csv")
            best_log.plot(output_dir / "best_result.png")
            print(f"当前最佳结果已更新: trial {best['trial']:02d}, "
                  f"RMS={best['error_rms']:.6g}, 图像={output_dir / 'best_result.png'}")
        except (ImportError, RuntimeError) as exc:
            print(f"最佳结果配置已更新，但绘图失败: {exc}", file=sys.stderr)

    print(f"实验目录: {output_dir}")
    print("每次实验前确认急停可用、工作区无人、仅第五关节允许运动。")
    for trial in range(1, args.trials + 1):
        trial_dir = output_dir / f"trial_{trial:02d}"
        trial_dir.mkdir()
        trial_cfg = trial_dir / "optimal_backstepping_torque.cfg"
        write_trial_config(base_text, params, trial_cfg)

        checked = subprocess.run(
            [str(binary), "--check-config", str(trial_cfg)],
            cwd=trial_dir, text=True, capture_output=True)
        (trial_dir / "check_output.txt").write_text(
            checked.stdout + checked.stderr, encoding="utf-8")
        if checked.returncode != 0:
            print(f"Trial {trial}: 配置检查失败，停止。")
            return 3

        print(f"\nTrial {trial}/{args.trials}: {json.dumps(params, ensure_ascii=False)}")
        token = input(f"确认现场安全后输入 RUN {trial}，其他输入停止: ").strip()
        if token != f"RUN {trial}":
            print("实验已由操作者停止。")
            break

        completed = subprocess.run(
            [str(binary), "--run", str(trial_cfg)],
            cwd=trial_dir, text=True, capture_output=True)
        console = completed.stdout + completed.stderr
        (trial_dir / "console.txt").write_text(console, encoding="utf-8")
        log_path = trial_dir / "optimal_backstepping_log.csv"
        if not log_path.is_file():
            print(f"Trial {trial}: 未生成日志，停止。请查看 {trial_dir / 'console.txt'}")
            return 4

        metrics = analyze_log(log_path, params)
        metrics["return_code"] = completed.returncode
        metrics["trial"] = trial
        metrics["params"] = dict(params)
        metrics["console_abort"] = "安全保护中止" in console
        results.append(metrics)
        (trial_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        update_best()
        if trial < args.trials:
            params = next_params(params, metrics)

    (output_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    valid = [item for item in results if not item["aborted"] and not item["console_abort"]]
    if valid:
        best = min(valid, key=lambda item: item["error_rms"])
        best_dir = output_dir / f"trial_{best['trial']:02d}"
        shutil.copy2(best_dir / "optimal_backstepping_torque.cfg",
                     output_dir / "best.cfg")
        print(f"最佳有效实验: trial {best['trial']:02d}, error RMS={best['error_rms']:.6g}")
        print(f"最佳配置: {output_dir / 'best.cfg'}")
    else:
        print("没有未中止的有效实验，未生成 best.cfg。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
