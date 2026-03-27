import numpy as np

from VisFly.utils.evaluate import TestBase
import os, sys
from typing import Optional
from matplotlib import pyplot as plt
from VisFly.utils.FigFashion.FigFashion import FigFon
import torch as th
import copy, cv2


class Test(TestBase):
    def __init__(self,
                 model,
                 name,
                 save_path: Optional[str] = None,
                 ):
        super(Test, self).__init__(model=model, name=name, save_path=save_path, )
        self.target_all = []

    def draw(self, names=None):
        state_data = th.stack(self.state_all).cpu()
        targets = th.stack(self.target_all)
        action = th.stack([th.tensor(a) for a in self.action_all]).cpu()
        t = np.stack(self.t)[:, 0]
        
        # 设置统一的线宽，方便调整
        lw = 2.5 

        for i in range(self.model.env.num_envs):
            fig = plt.figure(figsize=(7, 4))
            
            # 1. 位置 Position
            plt.subplot(2, 3, 1)
            plt.plot(t, state_data[:, i, 0:3], label=["x", "y", "z"], linewidth=lw)
            plt.legend()
            
            # 2. 姿态 Quaternion
            plt.subplot(2, 3, 2)
            plt.plot(t, state_data[:, i, 3:7], label=["w", "x", "y", "z"], linewidth=lw)
            plt.legend()
            
            # 3. 线速度 Linear Velocity
            plt.subplot(2, 3, 3)
            plt.plot(t, state_data[:, i, 7:10], label=["vx", "vy", "vz"], linewidth=lw)
            plt.legend()
            
            # 4. 角速度 Angular Velocity
            plt.subplot(2, 3, 4)
            plt.plot(t, state_data[:, i, 10:13], label=["wx", "wy", "wz"], linewidth=lw)
            plt.legend()
            
            # 5. 动作 Action (注意 action 长度通常比 t 少 1)
            plt.subplot(2, 3, 5)
            plt.plot(t[:-1], action[:, i, :], label=["a", "awx", "awy", "awz"], linewidth=lw)
            plt.legend()
            
            # 6. 目标距离 Target Distance
            plt.subplot(2, 3, 6)
            plt.plot(t, targets[:, i], label="target", linewidth=lw)
            plt.legend()
            
            plt.tight_layout()
            plt.show()

        plt.show()

        return [fig, ]
        # col_dis = np.array([collision["col_dis"] for collision in self.collision_all])
        # fig2, axes = FigFon.get_figure_axes(SubFigSize=(1, 1))
        # axes.plot(t, col_dis)
        # axes.set_xlabel("t/s")
        # axes.set_ylabel("closest distance/m")
        # print("rewards_sum: ", np_rewards)


    def test(
            self,
            policy=None,
            world=None,
            # model=None,
            is_fig: bool = True,
            is_video: bool = True,
            is_sub_video: bool = True,
            is_fig_save: bool = True,
            is_video_save: bool = True,
            render_kwargs={},

    ):
        if is_fig_save:
            if not is_fig:
                raise ValueError("is_fig_save must be True if is_fig is True")
        if is_video_save:
            if not is_video:
                raise ValueError("is_video_save must be True if is_video is True")
        if policy is None:
            policy = self.model.policy
        env = self.env

        # done_all = th.full((env.num_envs,), False)
        obs = env.reset(is_test=True)
        self._img_names = [name for name in obs.keys() if (("color" in name) or ("depth" in name) or ("semantic" in name))]
        self.obs_all.append(obs)
        self.state_all.append(env.state)
        self.info_all.append([{} for _ in range(env.num_envs)])
        self.t.append(env.t.clone())
        self.target_all.append((env.target - env.position).norm(dim=1))
        self.collision_all.append({"col_dis": env.collision_dis,
                                   "is_col": env.is_collision,
                                   "col_pt": env.collision_point})
        agent_index = [i for i in range(env.num_agent)]
        self.eq_r = []
        self.eq_l = []

        while True:
            with th.no_grad():
                action = policy.predict(obs, deterministic=True)
                if isinstance(action, tuple):
                    action = action[0]
                # obs, reward, done, info = env.step(action, is_test=True)
                if world is not None:
                    obs, reward, done, info = env.step(action, is_test=True, latent_func=world.step)
                else:
                    obs, reward, done, info = env.step(action, is_test=True)
                # = env.get_observation(), env.reward, env.done, env.info
                col_dis, is_col, col_pt = env.collision_dis, env.is_collision, env.collision_point
                state = env.state
                self.collision_all.append({"col_dis": col_dis, "is_col": is_col, "col_pt": col_pt})

            self.reward_all.append(reward)
            self.action_all.append(action)
            self.state_all.append(state)
            self.obs_all.append(obs)
            self.info_all.append(copy.deepcopy(info))
            self.target_all.append((env.target - env.position).norm(dim=1))
            self.t.append(env.t.clone())
            if env.visual:
                render_kwargs["points"] = th.atleast_2d(env.target)
                render_image = cv2.cvtColor(env.render(**render_kwargs)[0], cv2.COLOR_RGBA2RGB)
                self.render_image_all.append(render_image)
            # done_all[done] = True

            for i in reversed(agent_index):
                if done[i]:
                    self.eq_r.append(info[i]['episode']['r'].item())
                    self.eq_l.append(info[i]['episode']['l'].item())
                    agent_index.remove(i)

            if len(agent_index) == 0:
                break

        mean_r = th.as_tensor(self.eq_r, dtype=th.float32).mean().item()
        mean_l = th.as_tensor(self.eq_l, dtype=th.float32).mean().item()
        # print(f"Average Rewards:{mean_r}, Average Length:{mean_l}")

        if is_fig:
            figs = self.draw()
            if is_fig_save:
                for i, fig in enumerate(figs):
                    self.save_fig(fig, c=i)
        else:
            figs = []
        if is_video:
            self.play(is_sub_video=is_sub_video)
            if is_video_save:
                self.save_video()

        render_video = th.as_tensor(np.stack(self.render_image_all, axis=0)).unsqueeze(0) if len(self.render_image_all) > 0 else None
        return figs, render_video, mean_r, mean_l
