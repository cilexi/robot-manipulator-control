#!/usr/bin/env python3
"""Tune, run, analyze, and interactively plot nomal_backstepping."""

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Directly editable defaults. Command-line options temporarily override them.
DEFAULT_PARAMETERS = {
    "k1_5": 112.0,
    "k2_5": 187.0,
    "reference_amplitude5": 0.02,
    "reference_omega5": 0.523598775598,
    "reference_cycles": 2.0,
    "alpha_limit5": 3.0,
    "alpha_dot_limit5": 3.0,
    "virtual_accel_limit5": 50.0,
    "nominal_inertia5": 0.045,
    "iota": 1.0,
    "theta_initial": 0.05,
    "gamma_theta": 0.10,
    "sigma_theta": 0.10,
    "tau_limit5": 1.0,
    "enable_output_barrier": True,
    "enable_theta_adaptation": True,
}


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def rms(values):
    return math.sqrt(mean([value * value for value in values])) if values else float("nan")


def fraction(values):
    values = list(values)
    return mean([1.0 if value else 0.0 for value in values]) if values else 0.0


def absolute_error_integral(rows):
    """Integrate absolute tracking error with the logged timestamps."""
    if len(rows) < 2:
        return 0.0
    return sum(
        0.5 * (abs(previous["error5"]) + abs(current["error5"]))
        * (current["time"] - previous["time"])
        for previous, current in zip(rows, rows[1:])
    )


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    if not rows:
        raise ValueError("nomal_backstepping 日志没有数据行")
    return rows


def replace_scalar(text, key, value):
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*).*$", re.MULTILINE)
    updated, count = pattern.subn(rf"\g<1>{value:.10g}", text)
    if count != 1:
        raise ValueError(f"配置项 {key} 应出现一次，实际为 {count}")
    return updated


def replace_bool(text, key, value):
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*).*$", re.MULTILINE)
    updated, count = pattern.subn(
        rf"\g<1>{'true' if value else 'false'}", text
    )
    if count != 1:
        raise ValueError(f"配置项 {key} 应出现一次，实际为 {count}")
    return updated


def replace_joint5(text, key, value):
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)([^#\r\n]+)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"找不到五轴配置项 {key}")
    values = match.group(2).split()
    if len(values) != 6:
        raise ValueError(f"配置项 {key} 必须包含 6 个数")
    values[4] = f"{value:.10g}"
    return text[: match.start()] + match.group(1) + " ".join(values) + text[match.end() :]


def read_config_value(text, key):
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^#\r\n]+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"找不到配置项 {key}")
    return match.group(1).strip()


def parameters_from_config(text, fallback):
    """Recover plotting parameters from the cfg stored beside an old log."""
    params = dict(fallback)
    for key in params:
        if key == "tau_limit5":
            values = read_config_value(text, "tau_limit").split()
            if len(values) != 6:
                raise ValueError("tau_limit 必须包含 6 个数")
            params[key] = float(values[4])
        elif isinstance(params[key], bool):
            raw = read_config_value(text, key).lower()
            if raw not in {"true", "false", "1", "0"}:
                raise ValueError(f"{key} 不是有效布尔值")
            params[key] = raw in {"true", "1"}
        else:
            params[key] = float(read_config_value(text, key))
    return params


def effective_parameters(args):
    return {
        "k1_5": args.k1,
        "k2_5": args.k2,
        "reference_amplitude5": args.amplitude,
        "reference_omega5": args.omega,
        "reference_cycles": args.cycles,
        "alpha_limit5": args.alpha_limit,
        "alpha_dot_limit5": args.alpha_dot_limit,
        "virtual_accel_limit5": args.virtual_accel_limit,
        "nominal_inertia5": args.inertia5,
        "iota": args.iota,
        "theta_initial": args.theta_initial,
        "gamma_theta": args.gamma_theta,
        "sigma_theta": args.sigma_theta,
        "tau_limit5": args.tau_limit5,
        "enable_output_barrier": args.barrier == "on",
        "enable_theta_adaptation": args.theta_adaptation == "on",
    }


def print_parameters(params):
    descriptions = {
        "k1_5": "第一层反馈增益",
        "k2_5": "第二层反馈增益",
        "reference_amplitude5": "第五关节正弦幅值 rad",
        "reference_omega5": "第五关节角频率 rad/s",
        "reference_cycles": "完整正弦周期数",
        "alpha_limit5": "虚拟速度上限 rad/s",
        "alpha_dot_limit5": "虚拟速度导数上限 rad/s^2",
        "virtual_accel_limit5": "虚拟加速度上限 rad/s^2",
        "nominal_inertia5": "第五关节标称惯量",
        "iota": "鲁棒项设计参数",
        "theta_initial": "theta 初值",
        "gamma_theta": "theta 自适应增益",
        "sigma_theta": "theta 泄漏系数",
        "tau_limit5": "第五关节力矩上限 Nm",
        "enable_output_barrier": "输出障碍开关",
        "enable_theta_adaptation": "theta 自适应开关",
    }
    print("\n本次 nomal_backstepping 参数")
    print("-" * 78)
    for key, value in params.items():
        display = str(value) if isinstance(value, bool) else f"{value:.9g}"
        print(f"{key:<31}{display:>14}  {descriptions[key]}")
    print("-" * 78)
    print("长期默认值可在 nomal_backstepping.py 顶部 DEFAULT_PARAMETERS 中修改。\n")


def make_run_config(base_text, params):
    text = base_text
    for key, value in params.items():
        if isinstance(value, bool):
            text = replace_bool(text, key, value)
        elif key == "tau_limit5":
            text = replace_joint5(text, "tau_limit", value)
        else:
            if not math.isfinite(value):
                raise ValueError(f"{key} 必须是有限数字")
            text = replace_scalar(text, key, value)
    return text


def analyze(rows):
    active = [row for row in rows if row["sine_active"] != 0.0]
    positive = [row for row in active if row["dqd5"] > 1.0e-5]
    negative = [row for row in active if row["dqd5"] < -1.0e-5]

    def directional(source):
        return {
            "samples": len(source),
            "error_rms": rms([row["error5"] for row in source]),
            "error_mean": mean([row["error5"] for row in source]),
            "tau_mean": mean([row["tau_cmd5"] for row in source]),
        }

    held_drift = {}
    for joint in (1, 2, 3, 4, 6):
        initial = rows[0][f"q{joint}"]
        held_drift[str(joint)] = max(
            abs(row[f"q{joint}"] - initial) for row in rows
        )

    metrics = {
        "samples": len(rows),
        "duration_s": rows[-1]["time"],
        "absolute_error_integral_rad_s": absolute_error_integral(rows),
        "aborted": any(row["abort"] != 0.0 for row in rows),
        "error_rms": rms([row["error5"] for row in active]),
        "error_peak": max(abs(row["error5"]) for row in rows),
        "z1_rms": rms([row["z1"] for row in active]),
        "z2_rms": rms([row["z2"] for row in active]),
        "alpha_peak": max(abs(row["alpha5"]) for row in rows),
        "alpha_dot_peak": max(abs(row["alpha_dot5"]) for row in rows),
        "virtual_accel_peak": max(abs(row["virtual_accel5"]) for row in rows),
        "tau_raw_peak": max(abs(row["tau5_raw"]) for row in rows),
        "tau_cmd_peak": max(abs(row["tau_cmd5"]) for row in rows),
        "barrier_ratio_peak": max(abs(row["barrier_ratio5"]) for row in rows),
        "theta_initial_logged": rows[0]["theta_hat"],
        "theta_final": rows[-1]["theta_hat"],
        "theta_peak": max(row["theta_hat"] for row in rows),
        "alpha_saturation_fraction": fraction(
            row["alpha_saturated"] != 0.0 for row in rows
        ),
        "alpha_dot_saturation_fraction": fraction(
            row["alpha_dot_saturated"] != 0.0 for row in rows
        ),
        "virtual_accel_saturation_fraction": fraction(
            row["virtual_accel_saturated"] != 0.0 for row in rows
        ),
        "torque_amplitude_saturation_fraction": fraction(
            row["amplitude_saturated5"] != 0.0 for row in rows
        ),
        "torque_rate_saturation_fraction": fraction(
            row["rate_saturated5"] != 0.0 for row in rows
        ),
        "positive_reference_velocity": directional(positive),
        "negative_reference_velocity": directional(negative),
        "held_joint_drift_peak": held_drift,
    }
    metrics["directional_error_difference"] = abs(
        metrics["positive_reference_velocity"]["error_rms"]
        - metrics["negative_reference_velocity"]["error_rms"]
    )
    return metrics


def plot_interactive(rows, metrics, params, output_path=None):
    """Create the result figure, save it when requested, and show it."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("交互绘图需要 matplotlib") from exc

    time = [row["time"] for row in rows]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    fig.suptitle(
        "Joint 5 No-RL Backstepping\n"
        f"K1={params['k1_5']:.4g}, K2={params['k2_5']:.4g}, "
        f"A={params['reference_amplitude5']:.4g} rad, "
        f"omega={params['reference_omega5']:.4g} rad/s",
        fontsize=14,
    )

    axes[0, 0].plot(time, [row["q5"] for row in rows], label="q5")
    axes[0, 0].plot(time, [row["qd5"] for row in rows], "--", label="qd5")
    axes[0, 0].set_title("Trajectory and reference")
    axes[0, 0].set_ylabel("position (rad)")
    axes[0, 0].legend()

    axes[0, 1].plot(time, [row["error5"] for row in rows], label="q5-qd5")
    axes[0, 1].plot(time, [row["z1"] for row in rows], label="z1", alpha=0.8)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Physical and transformed errors")
    axes[0, 1].set_ylabel("error")
    axes[0, 1].legend()

    axes[1, 0].plot(time, [row["z2"] for row in rows], label="z2")
    axes[1, 0].plot(time, [row["dq5"] for row in rows], label="dq5")
    axes[1, 0].plot(time, [row["alpha5"] for row in rows], "--", label="alpha5")
    axes[1, 0].set_title("Second-layer state and virtual velocity")
    axes[1, 0].set_ylabel("rad/s")
    axes[1, 0].legend()

    axes[1, 1].plot(time, [row["alpha_dot5"] for row in rows], label="alpha_dot")
    axes[1, 1].plot(time, [row["term_z2_5"] for row in rows], label="-K2*z2")
    axes[1, 1].plot(time, [row["term_mu_5"] for row in rows], label="-mu_s")
    axes[1, 1].plot(time, [row["term_nu_5"] for row in rows], label="-nu_r")
    axes[1, 1].plot(
        time, [row["virtual_accel5"] for row in rows],
        label="v limited", linewidth=1.5
    )
    axes[1, 1].set_title("Virtual-acceleration components")
    axes[1, 1].set_ylabel("rad/s^2")
    axes[1, 1].legend(fontsize=8)

    axes[2, 0].plot(time, [row["tau5_raw"] for row in rows], label="backstepping tau5")
    axes[2, 0].plot(time, [row["tau_cmd5"] for row in rows], label="command tau5")
    axes[2, 0].axhline(
        params["tau_limit5"], color="red", linestyle="--", linewidth=0.9
    )
    axes[2, 0].axhline(
        -params["tau_limit5"], color="red", linestyle="--", linewidth=0.9
    )
    axes[2, 0].set_title("Joint-5 torque")
    axes[2, 0].set_xlabel("time (s)")
    axes[2, 0].set_ylabel("Nm")
    axes[2, 0].legend()

    axes[2, 1].plot(time, [row["theta_hat"] for row in rows], label="theta_hat")
    barrier_axis = axes[2, 1].twinx()
    barrier_axis.plot(
        time, [row["barrier_ratio5"] for row in rows],
        color="tab:orange", label="barrier ratio"
    )
    axes[2, 1].set_title("Robust parameter and barrier ratio")
    axes[2, 1].set_xlabel("time (s)")
    axes[2, 1].set_ylabel("theta_hat")
    barrier_axis.set_ylabel("barrier ratio")
    lines1, labels1 = axes[2, 1].get_legend_handles_labels()
    lines2, labels2 = barrier_axis.get_legend_handles_labels()
    axes[2, 1].legend(lines1 + lines2, labels1 + labels2, loc="best")

    axes[0, 0].text(
        0.02, 0.03,
        f"RMS={metrics['error_rms']:.4g} rad\n"
        f"direction diff={metrics['directional_error_difference']:.4g} rad\n"
        f"v saturation={100*metrics['virtual_accel_saturation_fraction']:.1f}%",
        transform=axes[0, 0].transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"结果可视化图片已保存: {output_path}")
    plt.show()


def stream_process(command, cwd):
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output.append(line)
    return process.wait(), "".join(output)


def build_parser(script_dir, repo):
    parser = argparse.ArgumentParser(
        description="运行、调参并显示第五关节 No-RL 反步实验"
    )
    parser.add_argument(
        "--config", type=Path, default=script_dir / "nomal_backstepping.cfg"
    )
    parser.add_argument(
        "--binary", type=Path, default=repo / "build/bin/nomal_backstepping"
    )
    parser.add_argument("--log", type=Path, help="只分析已有日志")
    parser.add_argument("--show-defaults", action="store_true")
    parser.add_argument("--k1", type=float, default=DEFAULT_PARAMETERS["k1_5"])
    parser.add_argument("--k2", type=float, default=DEFAULT_PARAMETERS["k2_5"])
    parser.add_argument(
        "--amplitude", type=float, default=DEFAULT_PARAMETERS["reference_amplitude5"]
    )
    parser.add_argument(
        "--omega", type=float, default=DEFAULT_PARAMETERS["reference_omega5"]
    )
    parser.add_argument(
        "--cycles", type=float, default=DEFAULT_PARAMETERS["reference_cycles"]
    )
    parser.add_argument(
        "--alpha-limit", type=float, default=DEFAULT_PARAMETERS["alpha_limit5"]
    )
    parser.add_argument(
        "--alpha-dot-limit", type=float,
        default=DEFAULT_PARAMETERS["alpha_dot_limit5"]
    )
    parser.add_argument(
        "--virtual-accel-limit", type=float,
        default=DEFAULT_PARAMETERS["virtual_accel_limit5"]
    )
    parser.add_argument(
        "--nominal-inertia5", "--inertia5", dest="inertia5", type=float,
        default=DEFAULT_PARAMETERS["nominal_inertia5"],
        help="第五关节标称惯量，即论文控制律中的 B 的逆"
    )
    parser.add_argument("--iota", type=float, default=DEFAULT_PARAMETERS["iota"])
    parser.add_argument(
        "--theta-initial", type=float, default=DEFAULT_PARAMETERS["theta_initial"]
    )
    parser.add_argument(
        "--gamma-theta", type=float, default=DEFAULT_PARAMETERS["gamma_theta"]
    )
    parser.add_argument(
        "--sigma-theta", type=float, default=DEFAULT_PARAMETERS["sigma_theta"]
    )
    parser.add_argument(
        "--tau-limit5", type=float, default=DEFAULT_PARAMETERS["tau_limit5"]
    )
    parser.add_argument(
        "--barrier", choices=("on", "off"),
        default="on" if DEFAULT_PARAMETERS["enable_output_barrier"] else "off"
    )
    parser.add_argument(
        "--theta-adaptation", choices=("on", "off"),
        default="on" if DEFAULT_PARAMETERS["enable_theta_adaptation"] else "off"
    )
    return parser


def main():
    script_dir = Path(__file__).resolve().parent
    repo = script_dir.parent.parent
    args = build_parser(script_dir, repo).parse_args()
    params = effective_parameters(args)
    if args.show_defaults:
        print_parameters(params)
        return 0

    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"找不到配置文件: {config_path}", file=sys.stderr)
        return 2
    base_text = config_path.read_text(encoding="utf-8")

    if args.log:
        log_path = args.log.resolve()
        if not log_path.is_file():
            print(f"找不到日志: {log_path}", file=sys.stderr)
            return 2
        sibling_cfg = log_path.with_name("arm5_backstepping.cfg")
        if sibling_cfg.is_file():
            params = parameters_from_config(
                sibling_cfg.read_text(encoding="utf-8"), params
            )
        print_parameters(params)
        rows = read_rows(log_path)
        metrics = analyze(rows)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        plot_interactive(
            rows, metrics, params,
            output_path=log_path.parent / "arm5_backstepping_result.png",
        )
        return 0

    print_parameters(params)
    binary = args.binary.resolve()
    if not binary.is_file():
        print(f"找不到可执行文件: {binary}，请先编译。", file=sys.stderr)
        return 2
    try:
        run_text = make_run_config(base_text, params)
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2

    run_dir = script_dir / "arm5_backstepping_runs" / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    run_dir.mkdir(parents=True)
    run_cfg = run_dir / "arm5_backstepping.cfg"
    run_cfg.write_text(run_text, encoding="utf-8")
    checked = subprocess.run(
        [str(binary), "--check-config", str(run_cfg)],
        cwd=run_dir,
        text=True,
        capture_output=True,
    )
    check_text = checked.stdout + checked.stderr
    (run_dir / "check_output.txt").write_text(check_text, encoding="utf-8")
    print(check_text, end="")
    if checked.returncode != 0:
        print("配置检查失败，未连接机械臂。", file=sys.stderr)
        return 3

    print(f"实验目录: {run_dir}")
    token = input("确认急停和现场安全后输入 RUN BS，其他输入取消: ").strip()
    if token != "RUN BS":
        print("实验已取消。")
        return 0
    return_code, console = stream_process(
        [str(binary), "--run", str(run_cfg)], run_dir
    )
    (run_dir / "console.txt").write_text(console, encoding="utf-8")
    log_path = run_dir / "arm5_backstepping_log.csv"
    if not log_path.is_file():
        print("未生成 arm5_backstepping_log.csv。", file=sys.stderr)
        return 4
    rows = read_rows(log_path)
    metrics = analyze(rows)
    metrics["return_code"] = return_code
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("即将打开交互式结果窗口；关闭窗口后脚本结束。")
    plot_interactive(
        rows, metrics, params,
        output_path=run_dir / "arm5_backstepping_result.png",
    )
    return 0 if return_code == 0 and not metrics["aborted"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
