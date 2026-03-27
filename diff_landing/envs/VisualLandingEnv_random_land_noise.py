import numpy as np
import torch as th
import cv2
from typing import Dict
from VisFly.utils.type import TensorDict

from .VisualLandingEnv_random_land import VisualLandingEnv


def _semantic_to_vis(img: np.ndarray) -> np.ndarray:
    """语义图 -> 黑白 BGR (背景黑, 目标白)"""
    if img.ndim == 3:
        img = img[0]
    vis = np.zeros((*img.shape, 3), dtype=np.uint8)
    vis[img > 0] = [255, 255, 255]   # 目标白
    vis[img == 0] = [0, 0, 0]      # 背景黑
    return vis


class VisualLandingEnvNoise(VisualLandingEnv):
    """VisualLandingEnv with observation noise for sim-to-real transfer."""

    def __init__(
            self,
            *args,
            observation_noise: bool = True,
            semantic_pixel_std: float = 0.08,
            semantic_dropout_prob: float = 0.02,
            velocity_std: float = 0.05,
            velocity_z_std_scale: float = 1.5,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.observation_noise = observation_noise
        self.semantic_pixel_std = semantic_pixel_std
        self.semantic_dropout_prob = semantic_dropout_prob
        self.velocity_std = velocity_std
        self.velocity_z_std_scale = velocity_z_std_scale
        self.debug_visualize_noise = False  # 设为 True 时在 get_observation 中弹出语义图

    def visualize_noise_debug(self, semantic: np.ndarray, env_idx: int = 0):
        """Debug 可视化：显示语义图，按任意键关闭。"""
        vis = _semantic_to_vis(semantic[env_idx])
        vis = cv2.resize(vis, (256, 192), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Noise Debug: Semantic", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _add_semantic_pixel_noise(self, pixel_tensor: th.Tensor) -> th.Tensor:
        """Add noise to pixel center (u_norm, v_norm) and simulate detection dropout."""
        if not self.observation_noise:
            return pixel_tensor
        noisy = pixel_tensor.clone()
        noise_uv = th.randn_like(noisy[:, :2], device=self.device) * self.semantic_pixel_std
        noisy[:, :2] = (noisy[:, :2] + noise_uv).clamp(-1.0, 1.0)
        dropout_mask = th.rand(noisy.shape[0], device=self.device) < self.semantic_dropout_prob
        noisy[dropout_mask, 2] = 0.0
        fp_mask = (noisy[:, 2] < 0.5) & (th.rand(noisy.shape[0], device=self.device) < self.semantic_dropout_prob / 2)
        noisy[fp_mask, 2] = 1.0
        return noisy

    def _add_velocity_noise(self, velocity: th.Tensor) -> th.Tensor:
        """Add Gaussian noise to velocity (body frame). Z-axis typically noisier."""
        if not self.observation_noise:
            return velocity
        noise = th.randn_like(velocity, device=self.device)
        noise[:, :2] *= self.velocity_std
        noise[:, 2] *= self.velocity_std * self.velocity_z_std_scale
        return velocity + noise

    def get_observation(self, indices=None) -> Dict:
        obs = super().get_observation(indices)
        if not self.observation_noise:
            return obs

        # Apply noise to state: pixel_tensor (first 3) and velocity (indices 7-9, body frame)
        state = obs["state"]
        pixel_noisy = self._add_semantic_pixel_noise(state[:, :3].clone())
        body_vel = self._world_to_body_velocity().detach() / 10
        velocity_noisy = self._add_velocity_noise(body_vel)
        state_noisy = th.cat([
            pixel_noisy,
            state[:, 3:7],
            velocity_noisy,
            state[:, 10:],
        ], dim=1)

        # 语义图不加噪声（真实世界语义较准确），仅对质心和线速度加噪声
        # 原代码: if getattr(self, "debug_visualize_noise", False): self.visualize_noise_debug(obs["semantic"].numpy())
        if getattr(self, "debug_visualize_noise", False) and "semantic" in obs:
            self.visualize_noise_debug(obs["semantic"].numpy())

        # 原代码: return TensorDict({"state": state_noisy, "semantic": obs["semantic"]})
        out = {"state": state_noisy}
        if "semantic" in obs:
            out["semantic"] = obs["semantic"]
        return TensorDict(out)
