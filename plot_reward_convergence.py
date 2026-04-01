import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_REWARD_TAG_CANDIDATES = [
    "metrics/episode_reward_total",   # PPO+MADDPG.py
    "episode/total_reward",           # maddpg.py
    "NO1_C & R/Reward",               # 旧版 baseline 命名
]


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x.copy()
    window = min(window, x.size)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smooth_valid = np.convolve(x, kernel, mode="valid")
    prefix = np.full(window - 1, np.nan, dtype=np.float64)
    return np.concatenate([prefix, smooth_valid], axis=0)


def detect_convergence_step(
    steps: np.ndarray,
    values: np.ndarray,
    window: int = 50,
    cv_threshold: float = 0.10,
) -> Optional[int]:
    if values.size < window or window <= 1:
        return None
    for i in range(window - 1, values.size):
        seg = values[i - window + 1: i + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size < max(2, window // 2):
            continue
        mean_v = float(np.mean(seg))
        std_v = float(np.std(seg))
        cv = std_v / (abs(mean_v) + 1e-8)
        if cv <= cv_threshold:
            return int(steps[i])
    return None


def save_csv(csv_path: str, steps: np.ndarray, raw: np.ndarray, smooth: np.ndarray):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "reward_raw", "reward_smooth"])
        for s, r, sm in zip(steps.tolist(), raw.tolist(), smooth.tolist()):
            writer.writerow([int(s), float(r), "" if np.isnan(sm) else float(sm)])


def _load_scalar_series(event_path: str, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从单个 TensorBoard 事件文件读取某个标量 tag，返回 (steps, values)。
    """
    ea = EventAccumulator(event_path, size_guidance={"scalars": 0})
    ea.Reload()
    events = ea.Scalars(tag)
    if not events:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    steps = np.asarray([e.step for e in events], dtype=np.int64)
    vals = np.asarray([e.value for e in events], dtype=np.float64)
    return steps, vals


def _find_event_files(logdir_or_file: str) -> List[str]:
    if os.path.isfile(logdir_or_file):
        return [logdir_or_file]

    all_files = []
    for root, _, files in os.walk(logdir_or_file):
        for name in files:
            if name.startswith("events.out.tfevents."):
                all_files.append(os.path.join(root, name))
    return sorted(all_files)


def _discover_reward_tag(event_files: List[str], user_tag: Optional[str]) -> str:
    if user_tag:
        return user_tag

    # 自动从候选 tag 中找到第一个可用项
    for tag in DEFAULT_REWARD_TAG_CANDIDATES:
        for f in event_files:
            try:
                ea = EventAccumulator(f, size_guidance={"scalars": 0})
                ea.Reload()
                if tag in ea.Tags().get("scalars", []):
                    return tag
            except Exception:
                continue

    # 兜底：挑第一个包含 reward 字样的标量 tag
    for f in event_files:
        try:
            ea = EventAccumulator(f, size_guidance={"scalars": 0})
            ea.Reload()
            for t in ea.Tags().get("scalars", []):
                if "reward" in t.lower():
                    return t
        except Exception:
            continue

    raise ValueError("未找到 reward 标量 tag，请手动指定 --tag。")


def _merge_series(event_files: List[str], tag: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    合并多个事件文件同一 tag 的数据，按 step 排序，重复 step 保留最后出现值。
    """
    steps_all = []
    vals_all = []
    for f in event_files:
        steps, vals = _load_scalar_series(f, tag)
        if steps.size == 0:
            continue
        steps_all.append(steps)
        vals_all.append(vals)

    if not steps_all:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    steps_cat = np.concatenate(steps_all)
    vals_cat = np.concatenate(vals_all)
    order = np.argsort(steps_cat, kind="stable")
    steps_cat = steps_cat[order]
    vals_cat = vals_cat[order]

    # 重复 step 保留最后一个值，避免多事件文件重叠导致曲线锯齿
    step_to_val = {}
    for s, v in zip(steps_cat.tolist(), vals_cat.tolist()):
        step_to_val[int(s)] = float(v)
    uniq_steps = np.array(sorted(step_to_val.keys()), dtype=np.int64)
    uniq_vals = np.array([step_to_val[int(s)] for s in uniq_steps], dtype=np.float64)
    return uniq_steps, uniq_vals


def _load_unified_curve_csv(csv_path: str) -> List[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    required = {"algorithm", "run_id", "episode", "reward"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"统一 CSV 缺少字段: {sorted(missing)}")
    return rows


def _build_algorithm_series_from_csv(
    rows: List[dict],
    algorithm_filter: Optional[List[str]],
    aggregate: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    groups: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    for r in rows:
        algo = str(r.get("algorithm", "")).strip()
        run_id = str(r.get("run_id", "")).strip()
        if not algo:
            continue
        if algorithm_filter and algo not in algorithm_filter:
            continue
        try:
            ep = int(float(r.get("episode", 0)))
            rew = float(r.get("reward", 0.0))
        except ValueError:
            continue
        groups.setdefault(algo, {}).setdefault(run_id, []).append((ep, rew))

    series_by_algo: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for algo, run_map in groups.items():
        # 每个 run 内按 episode 去重（保留最后值）
        cleaned_runs = {}
        for run_id, pts in run_map.items():
            step_to_val = {}
            for ep, rew in sorted(pts, key=lambda x: x[0]):
                step_to_val[ep] = rew
            eps = np.array(sorted(step_to_val.keys()), dtype=np.int64)
            vals = np.array([step_to_val[int(e)] for e in eps], dtype=np.float64)
            cleaned_runs[run_id] = (eps, vals)
        if not cleaned_runs:
            continue

        if aggregate == "latest_run":
            latest_run = sorted(cleaned_runs.keys())[-1]
            series_by_algo[algo] = cleaned_runs[latest_run]
        else:  # mean_over_runs
            all_eps = sorted({int(e) for eps, _ in cleaned_runs.values() for e in eps.tolist()})
            if not all_eps:
                continue
            mat = []
            for _, (eps, vals) in cleaned_runs.items():
                e2v = {int(e): float(v) for e, v in zip(eps.tolist(), vals.tolist())}
                mat.append([e2v.get(e, np.nan) for e in all_eps])
            arr = np.asarray(mat, dtype=np.float64)
            mean_vals = np.nanmean(arr, axis=0)
            series_by_algo[algo] = (np.asarray(all_eps, dtype=np.int64), mean_vals)
    return series_by_algo


def plot_from_unified_csv(args):
    rows = _load_unified_curve_csv(args.input_csv)
    algo_filter = None
    if args.algorithms:
        algo_filter = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    series_by_algo = _build_algorithm_series_from_csv(rows, algo_filter, args.aggregate)
    if not series_by_algo:
        raise ValueError("CSV 中没有可用于绘图的算法数据。")

    os.makedirs(args.save_dir, exist_ok=True)
    png_path = os.path.join(args.save_dir, f"{args.save_name}.png")
    csv_path = os.path.join(args.save_dir, f"{args.save_name}.csv")

    plt.figure(figsize=(10, 6))
    out_rows = [("algorithm", "step", "reward_raw", "reward_smooth", "convergence_step")]
    for algo in sorted(series_by_algo.keys()):
        steps, reward_raw = series_by_algo[algo]
        reward_smooth = moving_average(reward_raw, args.smooth_window)
        conv_step = detect_convergence_step(
            steps,
            reward_smooth,
            window=args.cv_window,
            cv_threshold=args.cv_threshold,
        )
        plt.plot(steps, reward_smooth, linewidth=2.0, label=f"{algo} (smooth)")
        if args.show_raw:
            plt.plot(steps, reward_raw, linewidth=0.9, alpha=0.25, label=f"{algo} (raw)")
        if conv_step is not None:
            plt.axvline(conv_step, linestyle="--", linewidth=1.1, alpha=0.65)
        conv_str = "" if conv_step is None else str(conv_step)
        for s, r, sm in zip(steps.tolist(), reward_raw.tolist(), reward_smooth.tolist()):
            out_rows.append((algo, int(s), float(r), "" if np.isnan(sm) else float(sm), conv_str))

    plt.title(args.title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out_rows)

    print("=== Reward Plot From Unified CSV Done ===")
    print(f"input_csv   : {args.input_csv}")
    print(f"aggregate   : {args.aggregate}")
    print(f"algorithms  : {', '.join(sorted(series_by_algo.keys()))}")
    print(f"png         : {png_path}")
    print(f"csv         : {csv_path}")


def plot_from_tensorboard(args):
    event_files = _find_event_files(args.logdir)
    if not event_files:
        raise FileNotFoundError(f"未在路径中找到事件文件: {args.logdir}")
    tag = _discover_reward_tag(event_files, args.tag)
    steps, reward_raw = _merge_series(event_files, tag)
    if steps.size == 0:
        raise ValueError(f"在事件文件中未找到 tag={tag} 的数据。")
    reward_smooth = moving_average(reward_raw, args.smooth_window)
    conv_step = detect_convergence_step(
        steps,
        reward_smooth,
        window=args.cv_window,
        cv_threshold=args.cv_threshold,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    png_path = os.path.join(args.save_dir, f"{args.save_name}.png")
    csv_path = os.path.join(args.save_dir, f"{args.save_name}.csv")

    plt.figure(figsize=(10, 6))
    plt.plot(steps, reward_raw, linewidth=1.0, alpha=0.35, label="raw reward")
    plt.plot(steps, reward_smooth, linewidth=2.0, label=f"moving average (w={args.smooth_window})")
    if conv_step is not None:
        plt.axvline(conv_step, linestyle="--", linewidth=1.5, label=f"converged @ step {conv_step}")
    plt.title(args.title)
    plt.xlabel("Episode / Step")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    save_csv(csv_path, steps, reward_raw, reward_smooth)

    print("=== Reward Plot From TensorBoard Done ===")
    print(f"logdir      : {args.logdir}")
    print(f"used tag    : {tag}")
    print(f"points      : {steps.size}")
    print(f"convergence : {conv_step if conv_step is not None else 'not found'}")
    print(f"png         : {png_path}")
    print(f"csv         : {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="统一绘制 reward 收敛曲线（支持统一 CSV 批量读图，也支持 TensorBoard）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["csv", "tb"],
        default="csv",
        help="csv: 从统一 CSV 读多算法；tb: 从 TensorBoard 日志读单算法",
    )
    # CSV 模式参数
    parser.add_argument(
        "--input-csv",
        type=str,
        default="D:/coding/analysis_outputs/metrics/training_curve_records.csv",
        help="统一收敛记录 CSV 路径",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=None,
        help="仅绘制指定算法，逗号分隔；不填则绘制全部",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        choices=["latest_run", "mean_over_runs"],
        default="latest_run",
        help="同一算法多次运行的聚合方式",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="CSV 模式下同时显示原始 reward 线",
    )
    # TensorBoard 模式参数
    parser.add_argument(
        "--logdir",
        type=str,
        default="D:/coding/result",
        help="TensorBoard 事件文件所在目录或单个事件文件路径",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="reward 的标量 tag；不填则自动探测",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=50,
        help="滑动平均窗口大小",
    )
    parser.add_argument(
        "--cv-window",
        type=int,
        default=50,
        help="收敛判定窗口大小（CV）",
    )
    parser.add_argument(
        "--cv-threshold",
        type=float,
        default=0.10,
        help="收敛判定阈值（CV<=threshold）",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Reward Convergence Curve",
        help="图标题",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="D:/coding/analysis_outputs/figures",
        help="输出目录（PNG + CSV）",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default="reward_convergence",
        help="输出文件名前缀",
    )
    args = parser.parse_args()

    if args.mode == "csv":
        plot_from_unified_csv(args)
    else:
        plot_from_tensorboard(args)


if __name__ == "__main__":
    main()
