from algorithm.SHAC import SHAC
import time
from collections import deque
from typing import Type, Optional, Dict, ClassVar, Any, Union

from stable_baselines3.common import logger
import os, sys
from gymnasium import spaces
import random

import numpy as np
import torch as th
from VisFly.utils.policies.td_policies import CnnPolicy, BasePolicy, MultiInputPolicy
from stable_baselines3.sac.policies import MultiInputPolicy as SACMultiInputPolicy
# from torch.distributions import
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_schedule_fn
from tqdm import tqdm
from stable_baselines3.common.utils import polyak_update, get_parameters_by_name
from VisFly.utils.algorithms.lr_scheduler import transfer_schedule
from VisFly.utils.test.debug import get_network_statistics, check_none_parameters
from copy import deepcopy
from stable_baselines3.common.utils import safe_mean

is_r_value = True
is_reset = True


class BPTT(SHAC):
    def __init__(
            self,
            *args,**kwargs
    ):

        super().__init__(
            *args, **kwargs
        )

    def _set_name(self):
        self.name = "BPTT"

    def train_actor(
            self,
            replay_data,
            total_timesteps: int=1e5,
    ):
        # assert self.H >= 1, "horizon must be greater than 1"
        if is_reset:
            self.dream_env.reset_agent_by_id(state=replay_data.states, reset_obs=replay_data.observations)
            self.dream_env.detach()
        actor_loss = 0.
        # pre_active = th.ones((self.actor_batch_size,), device=self.device, dtype=th.bool)
        discount_factor = th.ones((self.num_envs,), dtype=th.float32, device=self.device)
        episode_done = th.zeros((self.num_envs,), device=self.device, dtype=th.bool)

        dream_len = []
        for inner_step in range(self.H):
            # dream a horizon of experience
            obs = self.dream_env.get_observation()
            pre_obs = obs.clone()
            # iteration
            actions, log_prob, h = self.policy.actor.action_log_prob(pre_obs)
            clipped_actions = th.clip(
                actions, th.as_tensor(self.action_space.low, device=self.device), th.as_tensor(self.action_space.high, device=self.device)
            )

            # step
            obs, reward, done, info = self.dream_env.step(clipped_actions)
            for i in range(len(episode_done)):
                episode_done[i] = info[i]["episode_done"]

            reward, done = reward.to(self.device), done.to(self.device)

            # compute the loss
            actor_loss = actor_loss - reward * discount_factor
            # done_but_not_episode_end = ((done) | (inner_step == self.H-1))& ~episode_done

            discount_factor = discount_factor * self.gamma * ~done + done

        # update
        actor_loss = (actor_loss).mean() # average of value and accumlative rewards
        self.policy.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=False)
        th.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 1.)
        # record grad
        # get_network_statistics(self.actor, self._logger, is_record=pbar.n - previous_step >= self._dump_step)
        self.policy.actor.optimizer.step()
        # self.rollout_buffer.compute_returns()
        self.dream_env.detach()

        # update critic
        # for i in range(self.gradient_steps):
        #     values, _ = th.cat(self.policy.critic(self.rollout_buffer.obs, self.rollout_buffer.action), dim=-1).min(dim=-1)
        #     target = self.rollout_buffer.returns
        #     critic_loss = th.nn.functional.mse_loss(target, values)
        #     self.policy.critic.optimizer.zero_grad()
        #     critic_loss.backward()
        #     th.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 0.5)
        #     self.policy.critic.optimizer.step()
        #
        #     polyak_update(params=self.policy.critic.parameters(), target_params=self.policy.critic_target.parameters(), tau=self.tau)
        #     polyak_update(params=self.critic_batch_norm_stats, target_params=self.critic_batch_norm_stats_target, tau=1.)
        #
        # self.rollout_buffer.clear()

        self._logger.record("train/actor_loss", actor_loss.item())