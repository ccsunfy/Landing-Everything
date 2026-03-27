import copy
import io
import pathlib
import time
from collections import deque
from typing import Type, Optional, Dict, ClassVar, Any, Union, List, Iterable, Tuple

import os, sys
from gymnasium import spaces
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.vec_env import VecEnv

from VisFly.utils.policies.td_policies import BasePolicy, MultiInputPolicy
from stable_baselines3.common.type_aliases import Schedule, MaybeCallback, TrainFreq, TrainFrequencyUnit, RolloutReturn, GymEnv
from tqdm import tqdm
from stable_baselines3.common.utils import polyak_update, get_parameters_by_name, safe_mean, should_collect_more_steps
from torch.nn import functional as F
from VisFly.utils.algorithms.lr_scheduler import transfer_schedule
from VisFly.utils.test.debug import get_network_statistics
from copy import deepcopy
from VisFly.utils.common import set_seed

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.sac.sac import SAC, SelfSAC
from diff_landing.algorithms.common import FullDictReplayBuffer, DictReplayBuffer, compute_td_returns, DataBuffer3, SimpleRolloutBuffer, RequiresGrad
from diff_landing.algorithms.policy import Policy as SimplePolicy


cd = lambda x: x.clone().detach()
cdu = lambda x: x.clone().detach().cpu()


# from stable_baselines3.common.buffers import DictReplayBuffer

class BPTT(OffPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MultiInputPolicy": MultiInputPolicy,
        "SimplePolicy": SimplePolicy
    }
    observation_space: spaces.Space
    action_space: spaces.Space
    num_envs: int
    lr_schedule: Schedule

    def __init__(
            self,
            env,
            policy: Union[Type, str],
            train_env=None,
            policy_kwargs: Optional[Dict] = None,
            learning_rate: Union[float, Schedule] = 1e-3,
            comment: Optional[str] = None,
            save_path: Optional[str] = None,
            horizon: float = 1,
            tau: float = 0.005,
            gamma: float = 0.99,
            gradient_steps: int = 1,
            train_freq: Tuple[int, str] = (100, "step"),
            buffer_size: int = 1_000_000,
            batch_size: int = 256,
            pre_stop: float = 0.1,
            device: Optional[str] = "auto",
            seed: int = 42,
            ent_coef: Union[str, float] = 0.0001,
            target_update_interval: int = 1,
            target_entropy: Union[str, float] = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
            stats_window_size: Optional[int] = None,
            tensorboard_log: Optional[str] = None,
            verbose: int = 1,
            replay_buffer_class: Optional[Type[DictReplayBuffer]] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            actor_gradient_steps: Optional[int] = None,
            _init_setup_model: bool = True,
    ):
        root = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.save_path = f"{root}/saved" if save_path is None else save_path
        self.H = horizon
        self.tau = tau
        self.gamma = gamma
        self.comment = comment
        self.env = env
        self.env = train_env
        self.num_envs = env.num_envs

        self.learning_rate = learning_rate

        self.ent_coef = ent_coef

        self.pre_stop = pre_stop
        self._seed = seed
        self._set_seed()

        self.actor_gradient_steps = actor_gradient_steps
        self._actor_n_updates = 0

        stats_window_size = stats_window_size if stats_window_size else self.num_envs


        super().__init__(
            policy=policy,
            env=env,
            learning_rate=transfer_schedule(learning_rate),
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log if tensorboard_log is not None else self.save_path,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            batch_size=batch_size,
            buffer_size=buffer_size,
            use_sde_at_warmup=use_sde_at_warmup,
            tau=tau,
            gamma=gamma,
            gradient_steps=gradient_steps,
            train_freq=train_freq,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            support_multi_env=True,
            sde_support=False,
        )
        if _init_setup_model:
            self._setup_model()

        self._end_value = False

    def _set_seed(self):
        set_seed(self._seed)

    def enable_end_value(self):
        self._end_value = True

    def _set_name(self):
        self.name = "BPTT"

    def _setup_model(self):
        super()._setup_model()

        self._set_name()

        self.env.reset()
        if self.env:
            self.env.reset()
            self.env.set_requires_grad(True)

        self.policy.critic_bp = self.policy.critic

        self._create_save_path()

        self.rollout_buffer = SimpleRolloutBuffer(gamma=self.gamma)

        # self.actor_batch_norm_stats_target = get_parameters_by_name(self.policy.actor, ["running_"])
        self.actor_batch_norm_stats = get_parameters_by_name(self.policy.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.policy.critic, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.policy.critic_target, ["running_"])

    def _create_save_path(self):
        index = 1
        path = f"{self.save_path}/{self.name}_{self.comment}_{index}" if self.comment is not None \
            else f"{self.save_path}/{self.name}_{index}"
        while os.path.exists(path):
            index += 1
            path = f"{self.save_path}/{self.name}_{self.comment}_{index}" if self.comment is not None \
                else f"{self.save_path}/{self.name}_{index}"
        self.policy_save_path = path

    def train_actor(self, log_interval: Optional[int] = None) -> None:
        # assert self.H >= 1, "horizon must be greater than 1"
        ent_coef_loss = None
        self.env.detach()
        actor_loss = 0.
        # pre_active = th.ones((self.actor_batch_size,), device=self.device, dtype=th.bool)
        discount_factor = th.ones((self.env.num_envs,), dtype=th.float32, device=self.device)
        episode_done = th.zeros((self.env.num_envs,), device=self.device, dtype=th.bool)
        pre_start = th.ones((self.env.num_envs,), device=self.device, dtype=th.bool)  # is this step the first step
        for inner_step in range(self.H):
            # dream a horizon of experience
            obs = self.env.get_observation()
            pre_obs = obs.clone()
            # iteration
            actions, entropy = self.policy.actor.action_and_entropy(pre_obs)
            # step
            obs, reward, done, info = self.env.step(actions)
            for i in range(len(episode_done)):
                episode_done[i] = info[i]["episode_done"]

            # self.train_num_timesteps += self.env.num_envs
            # if done.any():
            #     self._update_train_info_buffer(info, done)

            reward, done = reward.to(self.device), done.to(self.device)

            # compute the temporal difference
            with th.no_grad():
                next_actions = self.policy.actor(obs)
                next_actions = next_actions if not isinstance(next_actions, tuple) else next_actions[0]
                next_values, _ = th.cat(self.policy.critic_target(obs.detach(), next_actions.detach()), dim=-1).min(dim=-1)

            # compute the loss
            actor_loss = actor_loss - reward * discount_factor - entropy * discount_factor * self.ent_coef
            done_but_not_episode_end = (done | (inner_step == self.H - 1)) & ~episode_done
            if done_but_not_episode_end.any() and self._end_value:
                actor_loss = actor_loss - \
                             next_values * discount_factor * self.gamma * done_but_not_episode_end

            discount_factor = discount_factor * self.gamma * ~done + done
            # pre_active = pre_active & ~done

            self.rollout_buffer.add(obs=cd(pre_obs),
                                    reward=cd(reward),
                                    action=cd(actions),
                                    next_obs=cd(obs),
                                    done=cd(done),
                                    episode_done=cd(episode_done),
                                    value=cd(next_values),
                                    )
            self._store_transition(self.replay_buffer,
                                   buffer_action=cdu(actions),
                                   new_obs=cdu(obs),
                                   reward=cdu(reward),
                                   dones=cdu(done),
                                   infos=info,
                                   )
            self._update_info_buffer(infos=info, dones=cdu(done))
            self.check_whether_dump(log_interval=log_interval, dones=cdu(done))

        # update
        actor_loss = actor_loss.mean()  # average of value and accumlative rewards
        self.policy.actor.optimizer.zero_grad()
        # actor_loss.backward(retain_graph=False)
        actor_loss.backward(retain_graph=True)
        th.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
        self.policy.actor.optimizer.step()
        # polyak_update(params=self.policy.actor.parameters(),
        #               target_params=self.policy.actor.parameters(), tau=self.tau)

        self.rollout_buffer.compute_returns()
        self.env.detach()

        # # update critic
        for i in range(self.gradient_steps):
            values, _ = th.cat(self.policy.critic(self.rollout_buffer.obs, self.rollout_buffer.action), dim=-1).min(dim=-1)
            target = self.rollout_buffer.returns
            critic_loss = th.nn.functional.mse_loss(target, values)
            self.policy.critic.optimizer.zero_grad()
            critic_loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 0.5)
            self.policy.critic.optimizer.step()

            polyak_update(params=self.policy.critic.parameters(), target_params=self.policy.critic_target.parameters(), tau=self.tau)
            polyak_update(params=self.critic_batch_norm_stats, target_params=self.critic_batch_norm_stats_target, tau=1.)

        self.rollout_buffer.clear()

        # 缓解训练逐渐变慢：长时间训练后 CUDA 显存易碎片化，后续分配变慢。
        # empty_cache() 释放 PyTorch 缓存的未用显存，减轻碎片。每 20 次更新执行一次以控制开销。
        if th.cuda.is_available() and (self._actor_n_updates + 1) % 20 == 0:
            th.cuda.empty_cache()

        self._logger.record("train/actor_loss", actor_loss.item())
        self._logger.record("train/critic_loss", critic_loss.item() if isinstance(critic_loss, th.Tensor) else critic_loss)
        self.logger.record("train/ent_coef_loss", (ent_coef_loss.item() if isinstance(ent_coef_loss, th.Tensor) else ent_coef_loss))

    def learn(
            self: SelfSAC,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: Optional[int] = None,
            tb_log_name: str = "Algorithm",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfSAC:
        log_interval = log_interval if log_interval else self.n_envs
        # add tqdm
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            self.name if self.comment is None else f"{self.name}_{self.comment}",
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"
        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

        with tqdm(total=total_timesteps, desc="Training Progress") as pbar:
            try:
                while self.num_timesteps < total_timesteps:

                    self.train(
                        gradient_steps=self.gradient_steps,
                        batch_size=self.batch_size,
                        log_interval=log_interval,
                    )

                    # Update the progress bar
                    pbar.update(self.num_timesteps - pbar.n)
            except KeyboardInterrupt:
                # print(f"Training interrupted by user, saving current model at {self.policy_save_path}")
                # self.save(self.policy_save_path)
                print("Training interrupted by user, stopping training.")

        callback.on_training_end()

        return self

    def train(self, gradient_steps: int, batch_size: int = 64, log_interval = None) -> None:
        actor_gradient_steps = gradient_steps if self.actor_gradient_steps is None else self.actor_gradient_steps
        self.policy.set_training_mode(True)

        optimizers = [self.policy.actor.optimizer, self.policy.critic.optimizer]
        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        for j in range(actor_gradient_steps):
            self.train_actor(log_interval=log_interval)
            self.num_timesteps += self.num_envs * self.H
            pass

        self._actor_n_updates += self.actor_gradient_steps
        self._n_updates += self.gradient_steps
        # self.num_timesteps += self.num_envs * self.H

        self.logger.record("train/actor_n_updates", self._actor_n_updates)
        self.logger.record("train/n_updates", self._n_updates)

        self.policy.set_training_mode(False)

    def save(
            self,
            path: Union[str, pathlib.Path, io.BufferedIOBase] = None,
            exclude: Optional[Iterable[str]] = None,
            include: Optional[Iterable[str]] = None,
    ) -> None:
        path = self.policy_save_path if path is None else path
        print(f"Saving model to {path}.zip")
        super().save(
            path,
            exclude=exclude,
            include=include,
        )

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self.policy.eval()
        action = self.policy.predict(observation, deterministic=deterministic)

        return action

    def check_whether_dump(self, dones, log_interval: Optional[int] = None) -> None:
        for idx, done in enumerate(dones):
            if done:
                # Update stats
                # num_collected_episodes += 1
                self._episode_num += 1

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))

            if len(self.ep_info_buffer[0]["extra"]) >= 0:
                for key in self.ep_info_buffer[0]["extra"].keys():
                    self.logger.record(
                        f"rollout/ep_{key}_mean",
                        safe_mean(
                            [ep_info["extra"][key] for ep_info in self.ep_info_buffer]
                        ),
                    )
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _sample_action(
            self,
            learning_starts: int,
            action_noise: Optional[ActionNoise] = None,
            n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:

        action = self.predict(self._last_obs, deterministic=False)
        action = action.cpu().numpy() if not isinstance(action, tuple) else action[0]
        buffer_action = copy.deepcopy(action)
        return action, buffer_action
