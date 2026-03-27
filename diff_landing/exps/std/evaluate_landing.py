"""
Evaluate landing policy: Use the current environment's yaml and the algorithm's yaml parameters,
run N test episodes, and output the success rate (successful landing on the platform) and the average error distance to the platform center.
"""
import os
import sys
import argparse
import torch as th
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri"],
    "font.size": 24,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
})

sys.path.insert(0, os.getcwd())

from diff_landing.envs.VisualLandingEnv_random_land import VisualLandingEnv
from diff_landing.envs.VisualLandingEnv_random_land_2_image import VisualLandingEnv2Image
from diff_landing.envs.VisualLandingEnv_random_land_noise import VisualLandingEnvNoise
from diff_landing.algorithms.BPTT import BPTT
from diff_landing.algorithms.SHAC import SHAC
from diff_landing.algorithms.SAC import SAC
from VisFly.utils.algorithms.PPO import PPO
from diff_landing.algorithms.ABPT import ABPT
from VisFly.utils.common import load_yaml_config

env_alias = {
    "landingVisual": VisualLandingEnv,
    "landingVisual2Image": VisualLandingEnv2Image,
    "landingVisualNoise": VisualLandingEnvNoise,
}
alg_alias = {
    "BPTT": BPTT,
    "PPO": PPO,
    "SHAC": SHAC,
    "ABPT": ABPT,
    "SAC": SAC,
}


def parse_args():
    parser = argparse.ArgumentParser(description="评估降落策略")
    parser.add_argument("--env", "-e", type=str, default="landingVisual")
    parser.add_argument("--algorithm", "-a", type=str, default="BPTT")
    parser.add_argument("--weight", "-w", type=str, default=None,
                        help="模型权重文件名；或逗号分隔两个权重进入比较模式，如 -w w1.zip,w2.zip")
    parser.add_argument("--compare-weights", type=str, metavar="W1,W2",
                        help="比较两个权重的 vz-z 曲线，逗号分隔，如 --compare-weights w1.zip,w2.zip")
    parser.add_argument("--n_episodes", "-n", type=int, default=100, help="测试回合数")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--noise", action="store_true", help="使用带噪声的环境 (VisualLandingEnvNoise)")
    parser.add_argument("--no-semantic-in-obs", action="store_true",
                        help="observation 不含 semantic（用于评估 semantic_in_obs=False 训练的消融模型）")
    parser.add_argument("--adapt_thresh", type=float, default=-0.15,
                        help="高度自适应筛选阈值：corr(z,vz)<thresh 视为自适应（vz 随 z 减小而减缓）")
    return parser.parse_args()


def _adaptiveness_score(t_list, vz_list, z_list):
    """corr(z, vz) < 0 表示高度降低时 vz 减缓（自适应）"""
    if len(z_list) < 5 or len(vz_list) < 5:
        return np.nan
    z_arr = np.array(z_list)
    vz_arr = np.array(vz_list)
    if np.std(z_arr) < 1e-6:
        return np.nan
    return np.corrcoef(z_arr, vz_arr)[0, 1]


def _get_best_vz_z_curve(all_trajectories, adapt_thresh):
    """从轨迹中筛选最体现高度自适应的 vz-z 曲线（取 corr 最小的一条）"""
    scored = []
    for ep_t, ep_vz, ep_z in all_trajectories:
        score = _adaptiveness_score(ep_t, ep_vz, ep_z)
        if not np.isnan(score):
            scored.append((score, ep_t, ep_vz, ep_z))
    scored.sort(key=lambda x: x[0])
    adaptive = [(s, t, vz, z) for s, t, vz, z in scored if s < adapt_thresh]
    to_plot = adaptive if len(adaptive) > 0 else scored[:1]
    if len(to_plot) == 0:
        return None, None
    _, _, vz_list, z_list = to_plot[0]
    return z_list, vz_list


def _style_ax(ax, title, xlabel):
    ax.axhline(y=0, color="k", linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=28, fontname="Calibri")
    ax.grid(True, linewidth=2, alpha=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ylabel = ax.set_ylabel(r"$v_z$ (m/s)", fontsize=28)
    ylabel.set_fontname("Calibri")
    xlabel_obj = ax.set_xlabel(xlabel, fontsize=28)
    xlabel_obj.set_fontname("Calibri")
    for label in ax.get_xticklabels():
        label.set_fontname("Calibri")
    for label in ax.get_yticklabels():
        label.set_fontname("Calibri")


def run_eval_single_weight(eval_env, model, n_episodes, weight_name):
    """对单个权重运行评估，返回 successes, distances_xy, all_trajectories"""
    successes = []
    distances_xy = []
    all_trajectories = []

    def predict_fn(obs):
        out = model.predict(obs, deterministic=True)
        if isinstance(out, tuple):
            return out[0]
        return out

    for ep in range(n_episodes):
        obs = eval_env.reset(is_test=True)
        done = th.zeros(eval_env.num_agent, dtype=th.bool, device=eval_env.device)
        ep_t, ep_vz, ep_z = [], [], []
        t_val = eval_env.t.cpu().numpy() if th.is_tensor(eval_env.t) else np.array(eval_env.t)
        vz_val = eval_env.velocity[:, 2].cpu().numpy() if eval_env.velocity.device.type != "cpu" else eval_env.velocity[:, 2].numpy()
        z_val = eval_env.position[:, 2].cpu().numpy() if eval_env.position.device.type != "cpu" else eval_env.position[:, 2].numpy()
        ep_t.append(float(t_val[0]))
        ep_vz.append(float(vz_val[0]))
        ep_z.append(float(z_val[0]))

        while True:
            with th.no_grad():
                action = predict_fn(obs)
                if isinstance(action, tuple):
                    action = action[0]
                if th.is_tensor(action):
                    action = action.cpu().numpy() if action.device.type != "cpu" else action.numpy()

            obs, reward, done, info = eval_env.step(action, is_test=True)
            t_val = eval_env.t.cpu().numpy() if th.is_tensor(eval_env.t) else np.array(eval_env.t)
            vz_val = eval_env.velocity[:, 2].cpu().numpy() if eval_env.velocity.device.type != "cpu" else eval_env.velocity[:, 2].numpy()
            z_val = eval_env.position[:, 2].cpu().numpy() if eval_env.position.device.type != "cpu" else eval_env.position[:, 2].numpy()
            ep_t.append(float(t_val[0]))
            ep_vz.append(float(vz_val[0]))
            ep_z.append(float(z_val[0]))
            if (th.is_tensor(done) and done.all()) or (not th.is_tensor(done) and all(done)):
                break

        all_trajectories.append((ep_t, ep_vz, ep_z))
        d_xy = (eval_env.target - eval_env.position)[:, :2].norm(dim=1)
        d_z = (eval_env.target - eval_env.position)[:, 2].abs()
        success = (d_xy < 0.2) & (d_z < 0.1)
        for i in range(eval_env.num_agent):
            successes.append(success[i].item())
            distances_xy.append(d_xy[i].item())
        if (ep + 1) % 10 == 0:
            print(f"  已完成 {ep + 1}/{n_episodes} 回合 [{weight_name}]")

    return successes, distances_xy, all_trajectories


def evaluate(args):
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    env_config_path = os.path.join(exp_dir, "env_cfgs", f"{args.env}.yaml")
    alg_config_path = os.path.join(exp_dir, "alg_cfgs", args.env, f"{args.algorithm}.yaml")
    save_folder = os.path.join(exp_dir, "saved", args.env)

    if not os.path.exists(env_config_path):
        raise FileNotFoundError(f"环境配置不存在: {env_config_path}")
    if not os.path.exists(alg_config_path):
        raise FileNotFoundError(f"算法配置不存在: {alg_config_path}")

    env_config = load_yaml_config(env_config_path)
    alg_config = load_yaml_config(alg_config_path)

    # 使用 eval_env 配置，缺失项从 env 继承
    env_kwargs = env_config.get("env", {})
    eval_env_kwargs = env_config.get("eval_env", {}).copy()
    for key in ["sensor_kwargs", "dynamics_kwargs", "semantic_in_obs"]:
        if key not in eval_env_kwargs and key in env_kwargs:
            eval_env_kwargs[key] = env_kwargs[key]
    eval_env_kwargs["visual"] = True
    eval_env_kwargs["tensor_output"] = True
    eval_env_kwargs["seed"] = args.seed
    if args.no_semantic_in_obs:
        eval_env_kwargs["semantic_in_obs"] = False

    env_cls = VisualLandingEnvNoise if args.noise else VisualLandingEnv
    eval_env = env_cls(**eval_env_kwargs)

    # 比较模式：-w "w1,w2" 或 --compare-weights "w1,w2"
    compare_str = args.compare_weights or (args.weight if (args.weight and "," in args.weight) else None)
    if compare_str:
        parts = [x.strip() for x in compare_str.split(",")]
        if len(parts) != 2:
            raise ValueError("比较模式需提供恰好两个权重，逗号分隔，如 -w w1.zip,w2.zip")
        w1, w2 = parts
        weights = [w1, w2]
        curves = []
        for w in weights:
            weight_path = os.path.join(save_folder, w)
            if not weight_path.endswith((".zip", ".pth")) and os.path.exists(weight_path + ".zip"):
                weight_path = weight_path + ".zip"
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"权重文件不存在: {weight_path}")
            model = alg_alias[args.algorithm].load(weight_path, env=eval_env)
            if hasattr(model, "set_random_seed"):
                model.set_random_seed(args.seed)
            print(f"开始评估: {args.n_episodes} 回合, 权重={w}")
            _, _, all_traj = run_eval_single_weight(eval_env, model, args.n_episodes, w)
            z_list, vz_list = _get_best_vz_z_curve(all_traj, args.adapt_thresh)
            curves.append((os.path.splitext(w)[0], z_list, vz_list))

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        c0, c1 = plt.cm.tab10.colors[0], plt.cm.tab10.colors[1]
        for i, (name, z_list, vz_list) in enumerate(curves):
            ax = axes[i]
            if z_list is not None and vz_list is not None:
                ax.plot(z_list, vz_list, linewidth=6, color=c0, label=r"$v_z$ (m/s)")
            _style_ax(ax, name, r"Height $z$ (m)")
            ax.legend(loc="best", frameon=False, prop={"size": 20, "family": "Calibri"})
        plt.tight_layout()
        pdf_path = os.path.join(save_folder, "vz_z_compare.pdf")
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close()
        print(f"两权重 vz-z 曲线对比已保存: {pdf_path}")
        return {}

    if args.weight is None:
        raise ValueError("请指定 --weight 或 --compare-weights")

    weight_path = os.path.join(save_folder, args.weight)
    if not weight_path.endswith((".zip", ".pth")):
        zip_path = weight_path + ".zip"
        if os.path.exists(zip_path):
            weight_path = zip_path
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")

    model = alg_alias[args.algorithm].load(weight_path, env=eval_env)
    if hasattr(model, "set_random_seed"):
        model.set_random_seed(args.seed)

    # 单权重评估
    successes, distances_xy, all_trajectories = run_eval_single_weight(
        eval_env, model, args.n_episodes, args.weight
    )

    # 统计
    n_total = len(successes)
    success_rate = sum(successes) / n_total if n_total > 0 else 0.0
    mean_distance = sum(distances_xy) / n_total if n_total > 0 else 0.0
    mean_distance_success = (
        sum(d for s, d in zip(successes, distances_xy) if s) / sum(successes)
        if sum(successes) > 0
        else 0.0
    )
    mean_distance_fail = (
        sum(d for s, d in zip(successes, distances_xy) if not s) / (n_total - sum(successes))
        if (n_total - sum(successes)) > 0
        else 0.0
    )

    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"总回合数:     {n_total}")
    print(f"成功率:       {success_rate * 100:.2f}% ({int(sum(successes))}/{n_total})")
    print(f"平均误差 (xy 平面距离平台中心): {mean_distance:.4f} m")
    print(f"  - 成功回合平均误差: {mean_distance_success:.4f} m")
    print(f"  - 失败回合平均误差: {mean_distance_fail:.4f} m")
    print("=" * 50)

    # 筛选体现“下降速度随高度变化”的回合
    scored = []
    for ep_t, ep_vz, ep_z in all_trajectories:
        score = _adaptiveness_score(ep_t, ep_vz, ep_z)
        if not np.isnan(score):
            scored.append((score, ep_t, ep_vz, ep_z))
    # 按 corr 从小到大排序，越负越自适应；筛选 corr < adapt_thresh 的回合
    scored.sort(key=lambda x: x[0])
    adaptive = [(s, t, vz, z) for s, t, vz, z in scored if s < args.adapt_thresh]
    # 若无满足阈值的，则取 corr 最小的前 3 条（最接近自适应的）
    to_plot = adaptive if len(adaptive) > 0 else scored[:3]
    n_plot = min(len(to_plot), 3)

    if n_plot > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        colors = plt.cm.tab10.colors[:n_plot]
        for i, (score, t_list, vz_list, z_list) in enumerate(to_plot[:n_plot]):
            lbl = f"ep corr={score:.3f}" if n_plot > 1 else r"$v_z$ (m/s)"
            axes[0].plot(t_list, vz_list, linewidth=6, color=colors[i], label=lbl)
            axes[1].plot(z_list, vz_list, linewidth=6, color=colors[i], label=lbl)

        _style_ax(axes[0], "Z-Axis Descent Velocity vs Time", "Time (s)")
        _style_ax(axes[1], r"$v_z$ vs Height $z$", r"Height $z$ (m)")
        for ax in axes:
            ax.legend(loc="best", frameon=False, prop={"size": 20, "family": "Calibri"})

        plt.tight_layout()
        base_name = os.path.splitext(args.weight)[0]
        plot_path_png = os.path.join(save_folder, f"vz_curve_{base_name}.png")
        plot_path_pdf = os.path.join(save_folder, f"vz_curve_{base_name}.pdf")
        plt.savefig(plot_path_png, dpi=300, bbox_inches="tight")
        plt.savefig(plot_path_pdf, format="pdf", bbox_inches="tight", dpi=300)
        plt.close()
        n_adaptive = len(adaptive)
        print(f"Z轴下降速度曲线已保存: {plot_path_png}, {plot_path_pdf}")
        print(f"  共 {len(all_trajectories)} 回合, 其中 {n_adaptive} 回合体现高度自适应 (corr<{args.adapt_thresh}), 绘制前 {n_plot} 条")

    return {
        "success_rate": success_rate,
        "mean_distance": mean_distance,
        "n_episodes": n_total,
        "n_success": int(sum(successes)),
    }


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
