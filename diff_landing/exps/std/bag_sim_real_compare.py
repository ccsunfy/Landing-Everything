#!/usr/bin/env python3
"""
从 rosbag 提取真机 /bfctrl/local_odom 与 /mavros/imu/data，
在仿真 VisualLandingEnv 中用同一策略从近似初始状态闭环 rollout，
绘制「位置(相对起点)、世界系速度、机体角速度」真机 vs 仿真对比曲线。

依赖：rosbag、numpy、matplotlib、torch；需在 diff_imitation 下运行以 import diff_landing。

示例：
  cd /path/to/diff_imitation
  python diff_landing/exps/std/bag_sim_real_compare.py \\
    --bag /path/to/debug_test_325_cylinder.bag \\
    --weight diff_landing/exps/std/saved/landingVisual/BPTT_imu_pixel_center_reward_23_1.zip \\
    --out /tmp/bag_sim_real.pdf
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_EXP_STD = os.path.dirname(os.path.abspath(__file__))
_DIFF_LANDING_ROOT = os.path.abspath(os.path.join(_EXP_STD, "..", ".."))
_DIFF_IMITATION_ROOT = os.path.abspath(os.path.join(_DIFF_LANDING_ROOT, ".."))
if _DIFF_IMITATION_ROOT not in sys.path:
    sys.path.insert(0, _DIFF_IMITATION_ROOT)

DEFAULT_WEIGHT = os.path.join(
    _DIFF_IMITATION_ROOT,
    "diff_landing",
    "exps",
    "std",
    "saved",
    "landingVisual",
    "BPTT_imu_pixel_center_reward_23_1.zip",
)
DEFAULT_BAG = os.path.join(
    os.path.dirname(_DIFF_IMITATION_ROOT),
    "paper3_Landing_Everying",
    "debug_test_325_cylinder.bag",
)


def _import_rosbag():
    try:
        import rosbag  # type: ignore

        return rosbag
    except Exception as e:
        print(f"无法 import rosbag: {e}", file=sys.stderr)
        sys.exit(1)


def read_bag_odom_imu(bag_path: str, topic_odom: str, topic_imu: str):
    rosbag = _import_rosbag()
    ts_o, px, py, pz = [], [], [], []
    vx, vy, vz = [], [], []
    qw, qx, qy, qz = [], [], [], []
    ts_i, wx, wy, wz = [], [], [], []
    b = rosbag.Bag(bag_path, "r")
    try:
        for topic, msg, t in b.read_messages(topics=[topic_odom, topic_imu]):
            sec = t.to_sec()
            if topic == topic_odom:
                p = msg.pose.pose.position
                v = msg.twist.twist.linear
                q = msg.pose.pose.orientation
                ts_o.append(sec)
                px.append(p.x)
                py.append(p.y)
                pz.append(p.z)
                vx.append(v.x)
                vy.append(v.y)
                vz.append(v.z)
                qw.append(q.w)
                qx.append(q.x)
                qy.append(q.y)
                qz.append(q.z)
            else:
                w = msg.angular_velocity
                ts_i.append(sec)
                wx.append(w.x)
                wy.append(w.y)
                wz.append(w.z)
    finally:
        b.close()

    def _sort(ts, *arrays):
        ts = np.asarray(ts, dtype=np.float64)
        if len(ts) == 0:
            return ts, [np.asarray(a) for a in arrays]
        order = np.argsort(ts)
        ts = ts[order]
        out = [np.asarray(a)[order] for a in arrays]
        return ts, out

    ts_o, (px, py, pz, vx, vy, vz, qw, qx, qy, qz) = _sort(
        ts_o, px, py, pz, vx, vy, vz, qw, qx, qy, qz
    )
    ts_i, (wx, wy, wz) = _sort(ts_i, wx, wy, wz)
    quat = np.stack([qw, qx, qy, qz], axis=1)
    return (
        ts_o,
        np.stack([px, py, pz], axis=1),
        np.stack([vx, vy, vz], axis=1),
        quat,
        ts_i,
        np.stack([wx, wy, wz], axis=1),
    )


def interp_on_grid(t_src, y_src, t_grid):
    """y_src: (N,) or (N, K)"""
    y_src = np.asarray(y_src, dtype=np.float64)
    if y_src.ndim == 1:
        return np.interp(t_grid, t_src, y_src)
    return np.stack([np.interp(t_grid, t_src, y_src[:, j]) for j in range(y_src.shape[1])], axis=1)


def load_env_and_policy(weight_zip: str, env_name: str, seed: int):
    import torch as th

    from VisFly.utils.common import load_yaml_config
    from diff_landing.algorithms.BPTT import BPTT
    from diff_landing.envs.VisualLandingEnv_random_land import VisualLandingEnv

    exp_std = os.path.dirname(os.path.abspath(__file__))
    env_cfg = load_yaml_config(os.path.join(exp_std, "env_cfgs", f"{env_name}.yaml"))
    env_kwargs = env_cfg.get("eval_env", {}).copy()
    for key in ["sensor_kwargs", "dynamics_kwargs", "semantic_in_obs", "semantic_normalize"]:
        if key not in env_kwargs and key in env_cfg.get("env", {}):
            env_kwargs[key] = env_cfg["env"][key]
    env_kwargs["visual"] = True
    env_kwargs["tensor_output"] = True
    env_kwargs["num_agent_per_scene"] = 1
    env_kwargs["num_scene"] = 1
    env_kwargs["seed"] = seed
    eval_env = VisualLandingEnv(**env_kwargs)
    model = BPTT.load(weight_zip, env=eval_env)
    return eval_env, model


def sim_rollout(eval_env, model, pos0, quat_wxyz, vel0, omega0, num_steps: int):
    """quat [w,x,y,z], 与 env 设备一致。"""
    import torch as th

    device = eval_env.device
    pos = th.as_tensor(pos0, dtype=th.float32, device=device).reshape(1, 3)
    ori = th.as_tensor(quat_wxyz, dtype=th.float32, device=device).reshape(1, 4)
    vel = th.as_tensor(vel0, dtype=th.float32, device=device).reshape(1, 3)
    ori_vel = th.as_tensor(omega0, dtype=th.float32, device=device).reshape(1, 3)

    eval_env.reset(is_test=True)
    agent_indices = th.tensor([0], device=device, dtype=th.long)
    obs = eval_env.reset_agent_by_id(agent_indices=agent_indices, state=(pos, ori, vel, ori_vel))

    def _predict(o):
        with th.no_grad():
            a = model.predict(o, deterministic=True)
            if isinstance(a, tuple):
                a = a[0]
            if not th.is_tensor(a):
                a = th.as_tensor(a, device=device)
            return a

    traj_p = []
    traj_v = []
    traj_w = []
    for _ in range(num_steps):
        traj_p.append(eval_env.position[0].detach().cpu().numpy().copy())
        traj_v.append(eval_env.velocity[0].detach().cpu().numpy().copy())
        traj_w.append(eval_env.angular_velocity[0].detach().cpu().numpy().copy())
        action = _predict(obs)
        obs, _, done, _ = eval_env.step(action, is_test=True)
        if (th.is_tensor(done) and done.all()) or (not th.is_tensor(done) and all(done)):
            break

    return (
        np.asarray(traj_p),
        np.asarray(traj_v),
        np.asarray(traj_w),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", type=str, default=DEFAULT_BAG)
    ap.add_argument("--weight", type=str, default=DEFAULT_WEIGHT)
    ap.add_argument("--env", type=str, default="landingVisual")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topic-odom", type=str, default="/bfctrl/local_odom")
    ap.add_argument("--topic-imu", type=str, default="/mavros/imu/data")
    ap.add_argument("--max-steps", type=int, default=400, help="仿真闭环步数（受 ctrl_dt 控制），至少为 1")
    ap.add_argument("--out", type=str, default="bag_sim_real_compare.pdf", help="输出 PDF 路径")
    args = ap.parse_args()
    if args.max_steps < 1:
        print("--max-steps 必须 >= 1，已改为 1", file=sys.stderr)
        args.max_steps = 1

    if not os.path.isfile(args.bag):
        print(f"找不到 bag: {args.bag}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.weight):
        print(f"找不到权重: {args.weight}", file=sys.stderr)
        sys.exit(1)

    ts_o, pos_o, vel_o, quat_o, ts_i, omega_i = read_bag_odom_imu(
        args.bag, args.topic_odom, args.topic_imu
    )
    if len(ts_o) < 5:
        print("odom 数据过少", file=sys.stderr)
        sys.exit(1)

    t0 = float(ts_o[0])
    t_end = float(ts_o[-1])
    duration = t_end - t0

    print("加载仿真与策略…")
    eval_env, model = load_env_and_policy(os.path.abspath(args.weight), args.env, args.seed)
    ctrl_dt = float(eval_env.envs.dynamics.ctrl_dt)

    num_steps = min(args.max_steps, max(1, int(duration / ctrl_dt)))
    t_grid = t0 + np.arange(num_steps, dtype=np.float64) * ctrl_dt
    if t_grid[-1] > t_end:
        num_steps = int(np.floor((t_end - t0) / ctrl_dt))
        num_steps = max(1, num_steps)
        t_grid = t0 + np.arange(num_steps, dtype=np.float64) * ctrl_dt

    pos_r = interp_on_grid(ts_o, pos_o, t_grid)
    vel_r = interp_on_grid(ts_o, vel_o, t_grid)
    omega_r = interp_on_grid(ts_i, omega_i, t_grid)

    p0 = pos_r[0]
    pos_r_rel = pos_r - p0

    quat0 = quat_o[0].astype(np.float32)
    vel0 = vel_o[0].astype(np.float32)
    p0_msg = pos_o[0].astype(np.float32)
    w0 = omega_r[0].astype(np.float32)

    print(f"闭环步数: {num_steps}, ctrl_dt={ctrl_dt:.4f}s, 时长≈{num_steps * ctrl_dt:.2f}s")
    print("仿真 rollout…")
    pos_s, vel_s, omega_s = sim_rollout(
        eval_env, model, p0_msg, quat0, vel0, w0, num_steps
    )

    n = min(len(pos_s), pos_r_rel.shape[0])
    if n < 1:
        print(
            "错误: 仿真未产生任何轨迹点（可能 --max-steps=0 或 rollout 失败）。",
            file=sys.stderr,
        )
        sys.exit(1)

    pos_s = pos_s[:n]
    vel_s = vel_s[:n]
    omega_s = omega_s[:n]
    pos_r_rel = pos_r_rel[:n]
    vel_r = vel_r[:n]
    omega_r = omega_r[:n]
    pos_s_rel = pos_s - pos_s[0]

    t_plot = np.arange(n, dtype=np.float64) * ctrl_dt

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 避免无中文字体时 PDF 整页空白；矢量字体内嵌便于预览
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    labels_p = [r"$p_x$", r"$p_y$", r"$p_z$"]
    labels_v = [r"$v_x$", r"$v_y$", r"$v_z$"]
    labels_w = [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]

    # 勿在 suptitle 后裸用 tight_layout()，否则子图高度可被压成 0，PDF 呈空白
    fig, axes = plt.subplots(
        3, 3, figsize=(12, 9), sharex=True, constrained_layout=True
    )
    fig.suptitle(
        "Bag (interp.) vs Sim (policy rollout)\n"
        "Position relative to first sample; vel./omega in each frame",
        fontsize=11,
    )

    for j in range(3):
        axes[0, j].plot(t_plot, pos_r_rel[:, j], label="bag", linewidth=1.2, color="C0")
        axes[0, j].plot(t_plot, pos_s_rel[:, j], "--", label="sim", linewidth=1.2, color="C1")
        axes[0, j].set_ylabel(labels_p[j] + " (m)")
        axes[0, j].grid(True, alpha=0.3)
        axes[0, j].legend(fontsize=8)

    for j in range(3):
        axes[1, j].plot(t_plot, vel_r[:, j], label="bag", linewidth=1.2, color="C0")
        axes[1, j].plot(t_plot, vel_s[:, j], "--", label="sim", linewidth=1.2, color="C1")
        axes[1, j].set_ylabel(labels_v[j] + " (m/s)")
        axes[1, j].grid(True, alpha=0.3)

    for j in range(3):
        axes[2, j].plot(t_plot, omega_r[:, j], label="bag", linewidth=1.2, color="C0")
        axes[2, j].plot(t_plot, omega_s[:, j], "--", label="sim", linewidth=1.2, color="C1")
        axes[2, j].set_ylabel(labels_w[j] + " (rad/s)")
        axes[2, j].set_xlabel("time (s)")
        axes[2, j].grid(True, alpha=0.3)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # bbox_inches='tight' 与 tight_layout 叠用易把画布裁没；此处已用 constrained_layout
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"已保存: {out_path} (n={n} points)")


if __name__ == "__main__":
    main()
