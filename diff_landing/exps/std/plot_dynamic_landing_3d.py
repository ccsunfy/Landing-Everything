"""
在 dynamicLanding 测试环境中回放策略，并绘制平台与无人机的 3D 轨迹。
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Calibri + 矢量 PDF；字号略小于评估图，避免 3D 子图拥挤
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri"],
    "font.size": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
})

sys.path.insert(0, os.getcwd())

from VisFly.utils.common import load_yaml_config
from diff_landing.envs.DynamicLandingEnv import DynamicLandingEnv
from diff_landing.algorithms.BPTT import BPTT
from VisFly.utils.algorithms.PPO import PPO
from diff_landing.algorithms.SHAC import SHAC
from diff_landing.algorithms.ABPT import ABPT
from diff_landing.algorithms.SAC import SAC


ALG_ALIAS = {
    "BPTT": BPTT,
    "PPO": PPO,
    "SHAC": SHAC,
    "ABPT": ABPT,
    "SAC": SAC,
}


def parse_args():
    parser = argparse.ArgumentParser(description="绘制动态降落 3D 轨迹")
    parser.add_argument("--algorithm", "-a", type=str, default="BPTT")
    parser.add_argument("--env", "-e", type=str, default="dynamicLanding")
    parser.add_argument(
        "--weight",
        "-w",
        type=str,
        default="diff_landing/exps/std/saved/dynamicLanding/BPTT_dynamic_landing_curr_with_sr_0.55_0.6_0.7_s3_1",
        help="权重路径（支持绝对路径/相对路径；无后缀时自动尝试 .zip）",
    )
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument(
        "--platform-z-mode",
        type=str,
        default="ground",
        choices=["ground", "raw", "target_minus_offset"],
        help="平台轨迹 z 处理方式: ground=z固定为地面; raw=动态物体原始z; target_minus_offset=target减offset",
    )
    parser.add_argument("--ground-z", type=float, default=0.0, help="ground 模式的地面高度")
    parser.add_argument("--cmap", type=str, default="turbo", help="轨迹速度着色 colormap")
    parser.add_argument(
        "--out",
        type=str,
        default="diff_landing/exps/std/saved/dynamicLanding/dynamic_landing_3d_traj.png",
        help="输出图片路径",
    )
    return parser.parse_args()


def resolve_weight_path(weight_arg: str, exp_dir: str) -> str:
    candidates = []
    p = Path(weight_arg)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path(os.getcwd()) / weight_arg)
        candidates.append(Path(exp_dir) / "saved" / "dynamicLanding" / weight_arg)

    expanded = []
    for c in candidates:
        expanded.append(c)
        if c.suffix == "":
            expanded.append(Path(str(c) + ".zip"))

    for c in expanded:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        "未找到权重文件。尝试过:\n" + "\n".join(str(x) for x in expanded)
    )


def _set_3d_axis_style(ax, title: str, xlabel: str, ylabel: str, zlabel: str):
    """标题/轴标签与 reward 图同字体，字号适配 3D 画布密度。"""
    ax.set_title(title, fontsize=16, fontname="Calibri")
    xl = ax.set_xlabel(xlabel, fontsize=14)
    xl.set_fontname("Calibri")
    yl = ax.set_ylabel(ylabel, fontsize=14)
    yl.set_fontname("Calibri")
    zl = ax.set_zlabel(zlabel, fontsize=14)
    zl.set_fontname("Calibri")
    for axis in ("x", "y", "z"):
        ax.tick_params(axis=axis, labelsize=11)
    for label in ax.get_xticklabels():
        label.set_fontname("Calibri")
    for label in ax.get_yticklabels():
        label.set_fontname("Calibri")
    for label in ax.get_zticklabels():
        label.set_fontname("Calibri")
    ax.grid(True, linewidth=1.2, alpha=0.55)


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    half = max_range / 2.0
    ax.set_xlim3d([x_mid - half, x_mid + half])
    ax.set_ylim3d([y_mid - half, y_mid + half])
    ax.set_zlim3d([z_mid - half, z_mid + half])


def get_platform_position(
    env,
    z_mode: str = "ground",
    ground_z: float = 0.0,
) -> np.ndarray:
    """
    获取平台真实位置（优先使用动态物体位置）。
    若不可用，则用 target 减去 target_height_offset 近似平台表面位置。
    """
    scene_manager = getattr(getattr(env, "envs", None), "sceneManager", None)
    dyn_positions = getattr(scene_manager, "dynamic_object_position", None) if scene_manager is not None else None
    if dyn_positions and len(dyn_positions) > 0 and dyn_positions[0] is not None:
        p = np.asarray(dyn_positions[0], dtype=np.float32)
        if p.ndim == 2:
            p = p[0]
        if p.size >= 3:
            p = p[:3].copy()
            if z_mode == "ground":
                p[2] = np.float32(ground_z)
            return p

    # 回退：target 是着陆目标点（通常比平台中心高 target_height_offset）
    t = env.target[0].detach().cpu().numpy().astype(np.float32).copy()
    if z_mode == "ground":
        t[2] = np.float32(ground_z)
    elif z_mode == "target_minus_offset":
        t[2] -= float(getattr(env, "target_height_offset", 0.0))
    return t


def main():
    args = parse_args()
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    env_cfg_path = os.path.join(exp_dir, "env_cfgs", f"{args.env}.yaml")
    env_config = load_yaml_config(env_cfg_path)

    eval_env_kwargs = env_config.get("eval_env", {}).copy()
    for key in ["sensor_kwargs", "dynamics_kwargs", "semantic_in_obs"]:
        if key not in eval_env_kwargs and key in env_config.get("env", {}):
            eval_env_kwargs[key] = env_config["env"][key]
    eval_env_kwargs["visual"] = True
    eval_env_kwargs["tensor_output"] = True
    eval_env_kwargs["seed"] = args.seed

    env = DynamicLandingEnv(**eval_env_kwargs)

    weight_path = resolve_weight_path(args.weight, exp_dir)
    if args.algorithm not in ALG_ALIAS:
        raise ValueError(f"未知算法: {args.algorithm}, 可选: {list(ALG_ALIAS.keys())}")
    model = ALG_ALIAS[args.algorithm].load(weight_path, env=env)

    obs = env.reset(is_test=True)

    drone_xyz = []
    target_xyz = []
    drone_vz = []

    drone_xyz.append(env.position[0].detach().cpu().numpy().copy())
    target_xyz.append(
        get_platform_position(
            env,
            z_mode=args.platform_z_mode,
            ground_z=args.ground_z,
        )
    )
    drone_vz.append(float(env.velocity[0, 2].detach().cpu().item()))

    for _ in range(args.max_steps):
        with th.no_grad():
            action = model.predict(obs, deterministic=True)
            if isinstance(action, tuple):
                action = action[0]
            if th.is_tensor(action):
                action = action.detach().cpu().numpy()

        obs, _, done, _ = env.step(action, is_test=True)
        drone_xyz.append(env.position[0].detach().cpu().numpy().copy())
        target_xyz.append(
            get_platform_position(
                env,
                z_mode=args.platform_z_mode,
                ground_z=args.ground_z,
            )
        )
        drone_vz.append(float(env.velocity[0, 2].detach().cpu().item()))

        done_flag = bool(done.all().item()) if th.is_tensor(done) else bool(np.all(done))
        if done_flag:
            break

    drone_xyz = np.asarray(drone_xyz)
    target_xyz = np.asarray(target_xyz)
    drone_vz = np.asarray(drone_vz, dtype=np.float32)
    descent_speed = np.clip(-drone_vz, a_min=0.0, a_max=None)  # 向下速度为正

    # 记录原始起始高度（用于图中标注）
    drone_start_world = drone_xyz[0].copy()
    z0_world = float(drone_start_world[2])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 无人机轨迹按下降速度着色（每段颜色取相邻两点速度均值）
    if len(drone_xyz) >= 2:
        seg_points = drone_xyz.reshape(-1, 1, 3)
        segments = np.concatenate([seg_points[:-1], seg_points[1:]], axis=1)
        seg_speed = 0.5 * (descent_speed[:-1] + descent_speed[1:])
        vmin, vmax = float(seg_speed.min()), float(seg_speed.max())
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1e-6
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        lc = Line3DCollection(segments, cmap=args.cmap, norm=norm, linewidths=3.8)
        lc.set_array(seg_speed)
        ax.add_collection3d(lc)
        cbar = fig.colorbar(lc, ax=ax, pad=0.08, fraction=0.03)
        cbar.set_label(r"Descent speed $-v_z$ (m/s)", fontsize=12)
        cbar.ax.yaxis.label.set_fontname("Calibri")
        cbar.ax.tick_params(labelsize=10)
        for t in cbar.ax.get_yticklabels():
            t.set_fontname("Calibri")
    else:
        ax.plot(drone_xyz[:, 0], drone_xyz[:, 1], drone_xyz[:, 2], lw=3.8, c="#1f77b4", label="Drone")

    ax.plot(target_xyz[:, 0], target_xyz[:, 1], target_xyz[:, 2], lw=3.2, ls="--", c="#ff7f0e", label="Platform")

    ax.scatter(drone_xyz[0, 0], drone_xyz[0, 1], drone_xyz[0, 2], c="#1f77b4", s=60, marker="o", label="Drone start")
    ax.scatter(drone_xyz[-1, 0], drone_xyz[-1, 1], drone_xyz[-1, 2], c="#1f77b4", s=80, marker="x", label="Drone end")
    ax.scatter(target_xyz[0, 0], target_xyz[0, 1], target_xyz[0, 2], c="#ff7f0e", s=60, marker="o", label="Platform start")
    ax.scatter(target_xyz[-1, 0], target_xyz[-1, 1], target_xyz[-1, 2], c="#ff7f0e", s=80, marker="x", label="Platform end")

    _set_3d_axis_style(
        ax,
        title="Dynamic landing trajectory (3D)",
        xlabel=r"$x$ (m)",
        ylabel=r"$y$ (m)",
        zlabel=r"$z$ (m)",
    )
    ax.text2D(
        0.02,
        0.98,
        rf"Drone start height $z_0$ = {z0_world:.3f} m",
        transform=ax.transAxes,
        fontsize=12,
        fontname="Calibri",
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )
    ax.legend(loc="best", frameon=False, prop={"size": 11, "family": "Calibri"})
    set_axes_equal(ax)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] 权重: {weight_path}")
    print(f"[OK] 轨迹点数: drone={len(drone_xyz)}, platform={len(target_xyz)}")
    print(f"[OK] 无人机起始高度 z0: {z0_world:.4f} m")
    print(f"[OK] 3D轨迹图已保存: {out_path}")


if __name__ == "__main__":
    main()
