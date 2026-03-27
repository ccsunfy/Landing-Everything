import io
import pathlib
import time
from collections import deque
from typing import Type, Optional, Dict, ClassVar, Any, Union, List, Iterable, Tuple

from stable_baselines3.common import logger
import os, sys
from gymnasium import spaces
import numpy as np
import torch as th
from stable_baselines3.common.base_class import SelfBaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.vec_env import VecEnv

from VisFly.utils.policies.td_policies import BasePolicy, MultiInputPolicy
# from stable_baselines3.sac.policies import MultiInputPolicy as SAC_MultiInputPolicy
from stable_baselines3.common.type_aliases import Schedule, MaybeCallback, TrainFreq, TrainFrequencyUnit, RolloutReturn, GymEnv
from stable_baselines3.common.utils import get_schedule_fn, should_collect_more_steps
from tqdm import tqdm
from stable_baselines3.common.utils import polyak_update, get_parameters_by_name, safe_mean
from torch.nn import functional as F
from VisFly.utils.algorithms.lr_scheduler import transfer_schedule
from VisFly.utils.test.debug import get_network_statistics
from copy import deepcopy
from VisFly.utils.common import set_seed

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.sac.sac import SAC, SelfSAC
from diff_landing.algorithms.common import FullDictReplayBuffer, DictReplayBuffer, compute_td_returns, DataBuffer3, SimpleRolloutBuffer, RequiresGrad
from diff_landing.algorithms.policy import Policy as SimplePolicy
is_start_value = True
is_end_value = True
is_r_value = True
is_reset = True


class ABPT(SAC):
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
            env_class: Type[VecEnv] = None,
            env_kwargs: Optional[Dict[str, Any]] = None,
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
            ent_coef: Union[str, float] = "auto",
            target_update_interval: int = 1,
            target_entropy: Union[str, float] = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            verbose: int = 1,
            replay_buffer_class: Optional[Type[DictReplayBuffer]] = FullDictReplayBuffer,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            actor_batch_size: int = 1000,
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
        self.env_class = env_class
        self.env_kwargs = env_kwargs
        self.num_envs = env.num_envs

        self.learning_rate = learning_rate

        self.ent_coef = ent_coef

        self.pre_stop = pre_stop
        self._seed = seed
        self._set_seed()

        self.actor_gradient_steps = actor_gradient_steps
        self._actor_n_updates = 0
        self.actor_batch_size = actor_batch_size

        self.env_kwargs = env_kwargs

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
            target_entropy=target_entropy,
            target_update_interval=target_update_interval,
            tau=tau,
            gamma=gamma,
            gradient_steps=gradient_steps,
            train_freq=train_freq,
            ent_coef=ent_coef,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            _init_setup_model=_init_setup_model,
        )

    def _set_seed(self):
        set_seed(self._seed)

    def _setup_pred_env(self):
        # self.dream_env = deepcopy(self.env)
        # self.dream_env.requires_grad = True
        self.env_kwargs["num_scene"] = 1
        self.env_kwargs["num_agent_per_scene"] = self.actor_batch_size
        self.env_kwargs["requires_grad"] = True
        self.env_kwargs["visual"] = False
        self.dream_env = self.env.__class__(**self.env_kwargs)

    def _set_name(self):
        self.name = "ABPT"

    def _setup_model(self):
        super()._setup_model()
        # build a separate evaluation env so that it will not disturb the normal process of training
        # self.env.to(self.device)
        self._setup_pred_env()
        self._set_name()

        self.env.reset()
        self.dream_env.reset()

        self.policy.critic_bp = self.policy.critic

        self._create_save_path()

        self.rollout_buffer = SimpleRolloutBuffer(gamma=self.gamma)

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
        self.policy_save_path = path+".zip"

    def train_actor(
            self,
            replay_data,
            total_timesteps: int=1e5,
    ):
        # assert self.H >= 1, "horizon must be greater than 1"
        ent_coef_loss = None
        if is_reset:
            self.dream_env.reset_agent_by_id(state=replay_data.states, reset_obs=replay_data.observations)
            self.dream_env.detach()
        actor_loss = 0.
        # pre_active = th.ones((self.actor_batch_size,), device=self.device, dtype=th.bool)
        discount_factor = th.ones((self.num_envs,), dtype=th.float32, device=self.device)
        episode_done = th.zeros((self.num_envs,), device=self.device, dtype=th.bool)
        pre_start = th.ones((self.actor_batch_size,), device=self.device, dtype=th.bool)  # is this step the first step
        for inner_step in range(self.H):
            # dream a horizon of experience
            obs = self.dream_env.get_observation()
            pre_obs = obs.clone()
            # iteration
            actions, log_prob, h = self.policy.actor.action_log_prob(pre_obs)
            clipped_actions = th.clip(
                actions, th.as_tensor(self.action_space.low, device=self.device), th.as_tensor(self.action_space.high, device=self.device)
            )

            if inner_step == 0:
                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                    ent_coef = th.exp(self.log_ent_coef.detach())
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    # ent_coef_losses.append(ent_coef_loss.item())

                else:
                    ent_coef = self.ent_coef_tensor

                if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

            if pre_start.any():
                values_bp, _ = th.cat(self.policy.critic(obs, actions), dim=1).min(dim=1)  # retain the gradient
                values_bp = values_bp - ent_coef * log_prob
            # step
            obs, reward, done, info = self.dream_env.step(clipped_actions)
            for i in range(len(episode_done)):
                episode_done[i] = info[i]["episode_done"]

            reward, done = reward.to(self.device), done.to(self.device)

            # compute the temporal difference
            next_actions, next_log_prob, _ = self.policy.actor.action_log_prob(obs)
            next_actions = next_actions.clip(
                th.as_tensor(self.action_space.low, device=self.device), th.as_tensor(self.action_space.high, device=self.device)
            )
            next_values, _ = th.cat(self.policy.critic_target(obs.detach(), next_actions.detach()), dim=-1).min(dim=-1)
            next_values = next_values - ent_coef * next_log_prob

            # compute the loss
            actor_loss = actor_loss - reward * discount_factor
            done_but_not_episode_end = ((done) | (inner_step == self.H-1))& ~episode_done
            if done_but_not_episode_end.any() and is_end_value:
                actor_loss = actor_loss - \
                             next_values * discount_factor * self.gamma * done_but_not_episode_end
            if is_start_value and pre_start.any():
                # the next value of previous step thus divide lambda
                actor_loss = actor_loss - values_bp * pre_start

            discount_factor = discount_factor * self.gamma * ~done + done
            pre_start = done
            # pre_active = pre_active & ~done

            self.rollout_buffer.add(obs=pre_obs.clone().detach(),
                                    reward=reward.clone().detach(),
                                    action=clipped_actions.clone().detach(),
                                    next_obs=obs.clone().detach(),
                                    done=done.clone().detach(),
                                    episode_done=episode_done.clone().detach(),
                                    value=next_values.clone().detach()
                                    )

        # update
        actor_loss = (actor_loss).mean() / 2  # average of value and accumlative rewards
        self.policy.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=False)
        th.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
        # record grad
        # get_network_statistics(self.actor, self._logger, is_record=pbar.n - previous_step >= self._dump_step)
        self.policy.actor.optimizer.step()
        self.rollout_buffer.compute_returns()
        self.dream_env.detach()

        # update critic
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

        self._logger.record("train/actor_loss", actor_loss.item())
        self._logger.record("train/critic_loss", critic_loss.item() if isinstance(critic_loss, th.Tensor) else critic_loss)
        self.logger.record("train/ent_coef", (ent_coef.item()))
        self.logger.record("train/ent_coef_loss", (ent_coef_loss.item() if isinstance(ent_coef_loss, th.Tensor) else ent_coef_loss))


    def learn(
            self: SelfSAC,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 500,
            tb_log_name: str = "ABPT",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfSAC:
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

                    rollout = self.collect_rollouts(
                        self.env,
                        train_freq=self.train_freq,
                        action_noise=self.action_noise,
                        callback=callback,
                        learning_starts=self.learning_starts,
                        replay_buffer=self.replay_buffer,
                        log_interval=log_interval,
                    )

                    if not rollout.continue_training:
                        break

                    if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                        # If no `gradient_steps` is specified,
                        # do as many gradients steps as steps performed during the rollout
                        gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                        # Special case when the user passes `gradient_steps=0`
                        if gradient_steps > 0:
                            self.train(batch_size=self.batch_size,
                                       gradient_steps=self.gradient_steps,
                                       actor_gradient_steps=self.actor_gradient_steps)
                            # if self.num_timesteps >= 3e6:
                            #     self.train_actor()

                    # Update the progress bar
                    pbar.update(self.num_timesteps - pbar.n)
            except KeyboardInterrupt:
                print("Training interrupted by user, saving current model...")
                self.save(self.policy_save_path + "_cache")

        callback.on_training_end()

        return self

    def train(self, gradient_steps: int, actor_gradient_steps: int, batch_size: int = 64) -> None:
        actor_gradient_steps = gradient_steps if actor_gradient_steps is None else actor_gradient_steps
        self.policy.set_training_mode(True)

        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        for j in range(actor_gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size=self.actor_batch_size)
            self.train_actor(replay_data=replay_data)

        self._actor_n_updates += actor_gradient_steps
        self._n_updates += gradient_steps

        self.logger.record("train/actor_n_updates", self._actor_n_updates)
        self.logger.record("train/n_updates", self._n_updates)

        # record Q gradient compensation
        # import tensorboard as tb
        # from algorithm.common import extract_all_paras
        # w, b = extract_all_paras(self.policy.actor)
        # self.logger.output_formats[1].writer.add_image("debug/w",w.reshape(1,1,-1))
        # self.logger.output_formats[1].writer.add_image("debug/b",b.reshape(1,1,-1))

        # n = self.policy.actor.features_extractor.state_extractor[0]
        # i_num = len(n.weight[:,0])
        # for i in range(i_num):
        #     self.logger.record(f"debug/w{i}", n.weight[i,0].item())
        #     self.logger.record(f"debug/b{i}", n.bias[i].item())
        # self.logger.output_formats[1].writer.add_tensor("debug/w",n.weight[:,0].items(), self.num_timesteps)
        # self.logger.output_formats[1].writer.add_tensor("debug/b",n.bias, self.num_timesteps)

    def _store_transition(
            self,
            replay_buffer: FullDictReplayBuffer,
            buffer_action: np.ndarray,
            new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
            reward: np.ndarray,
            dones: np.ndarray,
            infos: List[Dict[str, Any]],
            states: np.ndarray,
            # extra: Dict[str, Any] = None,
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            next_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
            infos,
            states,
            # extra
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: FullDictReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            states = env.full_state.clone().detach()
            # extra = deepcopy(env.extra)
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos, states)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
                        # self.save_paras()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def save_paras(self):
        if hasattr(self, "para_i"):
            self.para_i += 1
        else:
            self.para_i = 0

        # save parameters
        if not os.path.exists(self.logger.dir):
            os.makedirs(self.logger.dir)
        th.save(self.policy.state_dict(),f"{self.logger.dir}/paras_{self.para_i}.pth")
    #
    def save(
            self,
            path: Union[str, pathlib.Path, io.BufferedIOBase] = None,
            exclude: Optional[Iterable[str]] = None,
            include: Optional[Iterable[str]] = None,
    ) -> None:
        path = self.save_path if path is None else path
        super().save(
            path,
            exclude=exclude,
            include=include,
        )

    # def _sample_action(
    #         self,
    #         learning_starts: int,
    #         action_noise: Optional[ActionNoise] = None,
    #         n_envs: int = 1,
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Sample an action according to the exploration policy.
    #     This is either done by sampling the probability distribution of the policy,
    #     or sampling a random action (from a uniform distribution over the action space)
    #     or by adding noise to the deterministic output.
    #
    #     :param action_noise: Action noise that will be used for exploration
    #         Required for deterministic policy (e.g. TD3). This can also be used
    #         in addition to the stochastic policy for SAC.
    #     :param learning_starts: Number of steps before learning for the warm-up phase.
    #     :param n_envs:
    #     :return: action to take in the environment
    #         and scaled action that will be stored in the replay buffer.
    #         The two differs when the action space is not normalized (bounds are not [-1, 1]).
    #     """
    #     # Select action randomly or according to policy
    #     if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
    #         # Warmup phase
    #         unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
    #     else:
    #         # Note: when using continuous actions,
    #         # we assume that the policy uses tanh to scale the action
    #         # We use non-deterministic action in the case of SAC, for TD3, it does not matter
    #         assert self._last_obs is not None, "self._last_obs was not set"
    #         unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
    #         # unscaled_action, _, h = self.predict(self._last_obs, deterministic=False)
    #         # self.env.latent = h
    #
    #     # Rescale the action from [low, high] to [-1, 1]
    #     if isinstance(self.action_space, spaces.Box):
    #         scaled_action = self.policy.scale_action(unscaled_action)
    #
    #         # Add noise to the action (improve exploration)
    #         if action_noise is not None:
    #             scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
    #
    #         # We store the scaled action in the buffer
    #         buffer_action = scaled_action
    #         action = self.policy.unscale_action(scaled_action)
    #     else:
    #         # Discrete case, no need to normalize or clip
    #         buffer_action = unscaled_action
    #         action = buffer_action
    #     return action, buffer_action
    #

    def save(
            self,
            path: Union[str, pathlib.Path, io.BufferedIOBase] = None,
            exclude: Optional[Iterable[str]] = None,
            include: Optional[Iterable[str]] = None,
    ) -> None:
        print(f"Model has been saved as {self.policy_save_path}")
        super().save(
            path if path is not None else self.policy_save_path,
            exclude=exclude,
            include=include,
        )

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
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)