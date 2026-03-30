import os, sys
import torch as th

sys.path.append(os.getcwd())

# Only environments and algorithms present in this repository (landingVisual / dynamicLanding + BPTT PPO SAC).
from diff_landing.envs.VisualLandingEnv_random_land import VisualLandingEnv
from diff_landing.envs.DynamicLandingEnv import DynamicLandingEnv
from diff_landing.algorithms.BPTT import BPTT
from VisFly.utils.algorithms.PPO import PPO
from diff_landing.algorithms.SAC import SAC
import argparse
from VisFly.utils.common import load_yaml_config
from pathlib import Path

th.autograd.set_detect_anomaly(False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments", add_help=False)
    parser.add_argument("--comment", "-c", type=str, default="std")
    parser.add_argument("--train", "-t", type=int, default=1)
    parser.add_argument("--algorithm", "-a", type=str, default="BPTT")
    parser.add_argument("--env", "-e", type=str, default="landingVisual")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--weight", "-w", type=str, default=None)
    parser.add_argument(
        "--env_cfg",
        type=str,
        default=None,
        help="Env cfg file name or path, e.g. dynamicLanding_s1",
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        default=None,
        help="Comma-separated env cfg names/paths",
    )
    parser.add_argument(
        "--stage_steps",
        type=str,
        default=None,
        help="Comma-separated timesteps per curriculum stage",
    )
    parser.add_argument(
        "--stage_success",
        type=str,
        default=None,
        help="Comma-separated success-rate targets per stage, e.g. 0.55,0.70,0.85",
    )
    parser.add_argument(
        "--check_steps",
        type=int,
        default=200000,
        help="Timesteps per curriculum success-rate check",
    )
    parser.add_argument(
        "--min_success_samples",
        type=int,
        default=100,
        help="Minimum success samples before checking stage promotion",
    )
    return parser


env_alias = {
    "landingVisual": VisualLandingEnv,
    "dynamicLanding": DynamicLandingEnv,
}

alg_alias = {
    "BPTT": BPTT,
    "PPO": PPO,
    "SAC": SAC,
}

args = parse_args().parse_args()
save_folder = os.path.dirname(os.path.abspath(sys.argv[0])) + f"/saved/{args.env}/"
config = load_yaml_config(
    os.path.dirname(os.path.abspath(__file__)) + f"/alg_cfgs/{args.env}/{args.algorithm}.yaml"
)


def resolve_env_cfg_path(env_cfg_arg):
    if env_cfg_arg is None:
        return os.path.dirname(os.path.abspath(__file__)) + f"/env_cfgs/{args.env}.yaml"
    if os.path.isabs(env_cfg_arg) or "/" in env_cfg_arg:
        return env_cfg_arg
    cfg_name = env_cfg_arg if env_cfg_arg.endswith(".yaml") else f"{env_cfg_arg}.yaml"
    return os.path.dirname(os.path.abspath(__file__)) + f"/env_cfgs/{cfg_name}"


env_cfg_path = resolve_env_cfg_path(args.env_cfg)
env_config = load_yaml_config(env_cfg_path)

if not args.train:
    env_config["eval_env"]["visual"] = True


def latest_saved_zip():
    matches = sorted(
        Path(save_folder).glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return str(matches[0]) if matches else None


if args.train:
    if args.curriculum:
        stage_cfgs = [s.strip() for s in args.curriculum.split(",") if s.strip()]
        if len(stage_cfgs) == 0:
            raise ValueError("--curriculum is empty")
        if args.stage_steps:
            stage_steps = [int(s.strip()) for s in args.stage_steps.split(",") if s.strip()]
            if len(stage_steps) != len(stage_cfgs):
                raise ValueError("--stage_steps length must equal --curriculum stages length")
        else:
            total = int(config["learn"]["total_timesteps"])
            base = total // len(stage_cfgs)
            stage_steps = [base for _ in stage_cfgs]
            stage_steps[-1] += total - base * len(stage_cfgs)

        success_targets = None
        if args.stage_success:
            success_targets = [float(s.strip()) for s in args.stage_success.split(",") if s.strip()]
            if len(success_targets) != len(stage_cfgs):
                raise ValueError("--stage_success length must equal --curriculum stages length")

        prev_weight = args.weight if args.weight is not None else None
        for i, (stage_cfg, stage_t) in enumerate(zip(stage_cfgs, stage_steps), start=1):
            stage_env_cfg = load_yaml_config(resolve_env_cfg_path(stage_cfg))
            print(f"[curriculum] Stage {i}/{len(stage_cfgs)} cfg={stage_cfg} steps={stage_t}")
            env = env_alias[args.env](**stage_env_cfg["env"])

            if prev_weight is None:
                model = alg_alias[args.algorithm](
                    env=env,
                    seed=args.seed,
                    comment=f"{args.comment}_s{i}",
                    save_path=save_folder,
                    **config["algorithm"],
                )
            else:
                weight_path = prev_weight if os.path.isabs(prev_weight) else save_folder + prev_weight
                model = alg_alias[args.algorithm].load(weight_path, env=env)
                if hasattr(model, "comment"):
                    model.comment = f"{args.comment}_s{i}"
                if hasattr(model, "_create_save_path"):
                    model._create_save_path()

            if success_targets is None:
                learn_kwargs = dict(config["learn"])
                learn_kwargs["total_timesteps"] = int(stage_t)
                model.learn(**learn_kwargs)
            else:
                target_sr = success_targets[i - 1]
                max_stage_steps = int(stage_t)
                stage_trained = 0
                reached = False
                while stage_trained < max_stage_steps:
                    chunk_steps = min(int(args.check_steps), max_stage_steps - stage_trained)
                    learn_kwargs = dict(config["learn"])
                    learn_kwargs["total_timesteps"] = int(chunk_steps)
                    learn_kwargs["reset_num_timesteps"] = False
                    model.learn(**learn_kwargs)
                    stage_trained += int(chunk_steps)

                    success_buf = getattr(model, "ep_success_buffer", None)
                    sample_n = len(success_buf) if success_buf is not None else 0
                    success_rate = (float(sum(success_buf)) / sample_n) if sample_n > 0 else 0.0
                    print(
                        f"[curriculum] Stage {i} progress {stage_trained}/{max_stage_steps} "
                        f"success_rate={success_rate:.3f} target={target_sr:.3f} samples={sample_n}"
                    )

                    if sample_n >= int(args.min_success_samples) and success_rate >= target_sr:
                        reached = True
                        print(f"[curriculum] Stage {i} promoted by success rate.")
                        break

                if not reached:
                    print(
                        f"[curriculum] Stage {i} reached max steps without meeting target success rate."
                    )
            model.save()

            saved_path = None
            if hasattr(model, "policy_save_path"):
                saved_path = model.policy_save_path + ".zip"
            if saved_path is None or (not os.path.exists(saved_path)):
                saved_path = latest_saved_zip()
            if saved_path is None:
                raise FileNotFoundError("Cannot locate saved checkpoint after curriculum stage.")
            prev_weight = saved_path
            try:
                env.close()
            except Exception:
                pass

        print("[curriculum] Training finished.")
        sys.exit(0)

    print("[run] Creating environment...")
    env = env_alias[args.env](**env_config["env"])
    print("[run] Environment created.")
    print(f"Algorithm: {args.algorithm}")
    print(f"Available algorithms: {list(alg_alias.keys())}")
    print(f"Selected algorithm type: {type(alg_alias[args.algorithm])}")
    print(f"Selected algorithm value: {alg_alias[args.algorithm]}")

    print("[run] Creating model...")
    model = alg_alias[args.algorithm](
        env=env,
        seed=args.seed,
        comment=args.comment,
        save_path=save_folder,
        **config["algorithm"],
    )
    print("[run] Model created.")

    if args.weight is not None:
        weight_path = args.weight if os.path.isabs(args.weight) else save_folder + args.weight
        model = alg_alias[args.algorithm].load(weight_path, env=env)

    print(f"\n[TensorBoard] Monitor training: tensorboard --logdir={save_folder}\n")

    model.learn(**config["learn"])
    model.save()

else:
    eval_env = env_alias[args.env](**env_config["eval_env"])
    if args.weight is None:
        weight_path = latest_saved_zip()
        if weight_path is None:
            raise FileNotFoundError(
                f"[run] args.weight is None, and no *.zip checkpoint was found under {save_folder}. "
                f"Please pass --weight/-w explicitly."
            )
        weight_name = os.path.basename(weight_path)
    else:
        weight_path = args.weight if os.path.isabs(args.weight) else save_folder + args.weight
        weight_name = args.weight

    model = alg_alias[args.algorithm].load(weight_path, env=eval_env)
    from diff_landing.exps.test.landing.test import Test as landing_test

    test_handle = landing_test(
        model=model,
        save_path=save_folder + "/test",
        name=weight_name,
    )
    test_handle.test(**config["test"])
