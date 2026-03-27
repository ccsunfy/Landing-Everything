import numpy as np
from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from typing import Union, Tuple, List, Optional, Dict
import torch as th
import cv2 as cv2
import apriltag
from habitat_sim import SensorType
from gymnasium import spaces
from scipy.ndimage import center_of_mass
from VisFly.utils.type import TensorDict
from collections import deque
import os
from datetime import datetime

class VisualLandingEnv(DroneGymEnvsBase):
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: dict = None,
            dynamics_kwargs: dict = {},
            scene_kwargs: dict = {},
            sensor_kwargs: list = [],
            device: str = "cpu",
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
            tensor_output: bool = False,
            save_rgb: bool = True,  # 是否保存RGB图像
            rgb_save_dir: str = "rgb_images",  # 保存目录
            blackout_height: float = 0.5,  # 新增：分割图变黑的高度阈值
            blackout_transition: float = 0.1,  # 新增：从正常到全黑的过渡区间
    ):
        sensor_kwargs = [{
            "sensor_type": SensorType.SEMANTIC,
            "uuid": "semantic",
            "resolution": [64, 64],
            "position": [0, 0, -0.05],
            "orientation": [-np.pi / 2, 0, 0]
        }
            # {
            #     "sensor_type": SensorType.COLOR,
            #     "uuid": "color",
            #     "resolution": [640, 480],
            #     "position": [0, 0, -0.1],
            #     "orientation": [-np.pi / 2, 0, 0]
            # }
            ]
        random_kwargs = {
            "state_generator":
                {
                    "class": "Uniform",
                    "kwargs": [
                        {"position": {"mean": [4., 0., 3.0], "half": [1.0, 1.0, 0.5]}},
                    ]
                }
        }
        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            sensor_kwargs=sensor_kwargs,
            scene_kwargs=scene_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,
        )
        self.observation_space["state"] = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor([2.0, 0., 0.2] if target is None else target).reshape(1,-1)
        self.centers = None
        self.previous_position = deque(maxlen=2) 
        self.pastAction = th.zeros((self.num_envs, 12))  
        self.previous_actions = deque(maxlen=4)  
        self.last_action = th.zeros((self.num_envs, 4)) 
        self.last_position = th.zeros((self.num_envs, 3))
        self.initial_heights = None  
        self.r_type = 2
        self.target_ori = th.tensor([1,0,0,0])
        self.success_radius = 0.03
        
        self.image_center = th.tensor([32, 32], device=self.device, dtype=th.float32)
        
        
        # 新增：分割图变黑相关参数
        self.blackout_height = blackout_height  # 分割图开始变黑的高度
        self.blackout_transition = blackout_transition  # 过渡区间
        self.min_blackout_height = blackout_height - blackout_transition  # 完全变黑的高度
        
        if self.save_rgb:
            os.makedirs(self.rgb_save_dir, exist_ok=True)
        
    def load_json(self, path):
        import json
        with open(path, 'r') as f:
            js = json.load(f)
        return js
    
    def reset(self, indices: Optional[Union[List[int], th.Tensor]] = None, is_test=False) -> TensorDict:
        obs = super().reset(indices)
        for i in range(self.envs.sceneManager.num_scene):
            path = self.envs.sceneManager.scenes[i].config.sim_cfg.scene_id
            js = self.load_json(path)

            translation = None
            for obj in js.get("object_instances", []):
                if "translation" in obj:
                    translation = obj["translation"]
                    translation = [-translation[2],-translation[0],translation[1]+0.3]  # plus height of the platform
                    break  
            if translation is None:
                translation = [2.0, 0.0, 0.2]  # default
                print(f"Warning: No translation found in {path}, using default: {translation}")
            self.target[i*self.num_agent_per_scene:(i+1)*self.num_agent_per_scene] = \
                th.tensor(translation, device=self.device).float().unsqueeze(0).repeat(self.num_agent_per_scene, 1)
        
        self.step_count = 0
        self.episode_count += 1
        
        return obs
    
    def _apply_blackout_effect(self, semantic_obs: np.ndarray) -> np.ndarray:
        """
        根据当前高度应用分割图变黑效果（支持多环境）
        当无人机高度低于blackout_height时，分割图逐渐变黑
        当高度低于min_blackout_height时，分割图完全变黑
        """
        if self.position is None or len(self.position) == 0:
            return semantic_obs
            
        # 获取所有环境的当前无人机高度
        current_heights = self.position[:, 2].detach().numpy()  # 所有环境的高度
        
        # 创建一个与输入相同形状的输出数组
        result_obs = np.zeros_like(semantic_obs)
        
        # 对每个环境单独处理
        for env_idx in range(len(current_heights)):
            current_height = current_heights[env_idx]
            
            # 如果高度高于变黑阈值，保持原始图像
            if current_height > self.blackout_height:
                result_obs[env_idx] = semantic_obs[env_idx]
                continue
                
            # 如果高度低于完全变黑阈值，使用全黑图像
            if current_height <= self.min_blackout_height:
                result_obs[env_idx] = np.zeros_like(semantic_obs[env_idx])
                continue
                
            # 在过渡区间内，线性插值
            transition_ratio = (current_height - self.min_blackout_height) / self.blackout_transition
            black_image = np.zeros_like(semantic_obs[env_idx])
            
            # 线性混合：transition_ratio=1时完全显示原图，=0时完全变黑
            mixed_image = semantic_obs[env_idx] * transition_ratio + black_image * (1 - transition_ratio)
            result_obs[env_idx] = mixed_image.astype(semantic_obs[env_idx].dtype)
        
        return result_obs
    
    def get_observation(
            self,
            indices=None
    ) -> Dict:
        state = th.hstack([
            self.position / self.max_sense_radius,
            self.orientation,
            self.velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)
        
        semantic_obs = self.sensor_obs["semantic"].astype(np.float32)
        
        # 应用分割图变黑效果（支持多环境）
        semantic_obs_with_blackout = self._apply_blackout_effect(semantic_obs)

        # if self.save_rgb:
        #     self._save_rgb_images()
            
        return TensorDict({
            "state": state,
            # 'color': self.sensor_obs["color"],
            "semantic": th.as_tensor(semantic_obs_with_blackout)
        })
    
    # def _save_rgb_images(self):
    #     try:
    #         rgb_imgs = self.sensor_obs["rgb"]
    #
    #         for env_idx in range(len(rgb_imgs)):
    #             rgb_img = rgb_imgs[env_idx]
                
    #             # 转换为numpy数组
    #             if isinstance(rgb_img, th.Tensor):
    #                 rgb_img = rgb_img.cpu().numpy()
                
    #             # 确保数据类型正确
    #             if rgb_img.dtype == np.float32 or rgb_img.dtype == np.float64:
    #                 if rgb_img.max() <= 1.0:
    #                     rgb_img = (rgb_img * 255).astype(np.uint8)
    #                 else:
    #                     rgb_img = rgb_img.astype(np.uint8)
                
    #             # 创建文件名并保存
    #             filename = f"ep_{self.episode_count:04d}_step_{self.step_count:06d}_env_{env_idx}.png"
    #             filepath = os.path.join(self.rgb_save_dir, filename)
                
    #             success = cv2.imwrite(filepath, rgb_img)
    #             if not success:
    #                 print(f"Failed to save RGB image: {filepath}")
                    
    #     except Exception as e:
    #         print(f"Error saving RGB images: {e}")
    
    # def get_success(self) -> th.Tensor:
    #     return (self.position - self.target).norm(dim=1) < self.success_radius

    def get_success(self) -> th.Tensor:
        d_xy = (self.target - self.position)[:, :2].norm(dim=1)
        d_z = (self.target - self.position)[:, 2].abs()
        return (d_xy < self.success_radius) & (d_z < 0.1)
    
    def get_reward(self) -> Dict[str, th.Tensor]:
        d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
        r_xy = -d_xy * 0.04
        
        v_l = (0.5 * self.position[:, 2] - 0).clip(min=0.05, max=0.5).clone().detach()
        r_z = 1/(1+(-v_l-self.velocity[:,-1]).abs()) * 0.1
        r_v = -self.velocity.norm(dim=1) * 0.01
        r_omega = (self.angular_velocity - 0).norm(dim=1) * -0.001
        r_ori = (self.orientation - self.target_ori).norm(dim=1) * -0.003
        r_s = 10.0 
        r_l = self.get_success() * r_s + self.failure * -1.0
        
        diff_r = r_xy + r_z + r_omega + r_ori
        disc_r = r_l
        reward = diff_r + disc_r
        
        return {
            "diff_r": diff_r, 
            "disc_r": disc_r, 
            "reward": reward, 
            "success": self.success.float(),
            "r_z": r_z
        }