import numpy as np
from VisFly.envs.base.droneGymEnv import DroneGymEnvsBase
from typing import Union, Tuple, List, Optional, Dict
import torch as th
# th.autograd.set_detect_anomaly(True)
import cv2 as cv2
import apriltag
from habitat_sim import SensorType
from gymnasium import spaces
from scipy.ndimage import center_of_mass
from VisFly.utils.type import TensorDict
from collections import deque
import os
from datetime import datetime

class VisualLandingEnv2Image(DroneGymEnvsBase):
    """
    单帧 semantic 输入：policy 观测为当前帧 (N, 1, H, W)，observation_space['semantic'].shape == (1, H, W)，
    CNN in_channels=1。上一帧仅缓存用于 state 中的 d_mask_area（mask 面积差分），不拼进图像通道。
    """
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
            save_rgb: bool = False,  
            rgb_save_dir: str = "rgb_images",  
            blackout_height: float = 0.6, 
            blackout_transition: float = 0.1,
            semantic_in_obs: bool = True,  # False: semantic 仅用于 reward（pixel_tensor），不进入 policy 观测；原代码无此参数
    ):
        sensor_kwargs = [{
            "sensor_type": SensorType.SEMANTIC,
            "uuid": "semantic",
            "resolution": [48, 64],
            "position": [0, 0, -0.03],
            "orientation": [-np.pi / 2, 0, 0]
        },
        #                  {
        #     "sensor_type": SensorType.COLOR,
        #     "uuid": "color",
        #     "resolution": [480, 640],
        #     "position": [0, 0, -0.05],
        #     "orientation": [-np.pi / 2, 0, 0]
        # }
            ]
        random_kwargs = {
            "state_generator":
                    {
                    "class": "Uniform",
                    "kwargs": [
                        {"position": {"mean": [1.5, 0.0, 2.0], "half": [0.2, 0.2, 0.2]}},
                    ]
                    }
                    # {
                    # "class": "Uniform",
                    # "kwargs": [
                    #     {"position": {"mean": [1.0, 0., 1.0], "half": [1.0, 1.0, 1.0]}},
                    # ]
                    # }
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
        # state: pixel(3) + quat(4) + body angular vel(3) + mask_area(1) + d_mask_area(1) + du(1) + dv(1) = 14
        # 其中 mask_area/d_mask_area/du/dv 用于在不显式输入线速度时提供视觉时序线索
        self.observation_space["state"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        # semantic: 单帧当前 mask，形状 (1, H, W)
        self._semantic_hw = tuple(sensor_kwargs[0]["resolution"])  # (H, W)
        self.semantic_in_obs = semantic_in_obs
        if not semantic_in_obs and "semantic" in self.observation_space.spaces:
            del self.observation_space.spaces["semantic"]
        # elif semantic_in_obs and "semantic" in self.observation_space.spaces:
        #     h, w = self._semantic_hw
        #     self.observation_space.spaces["semantic"] = spaces.Box(
        #         low=0.0, high=255.0, shape=(1, h, w), dtype=np.float32
        #     )
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
        self.success_radius = 0.1
        self.total_timesteps = 0
        self.pixel_tensor = th.zeros((self.num_envs, 3), device=self.device)
        # self.target_update_interval = 120000
        self.target_update_interval = 100000
        
        self.image_center = th.tensor([32, 32], device=self.device, dtype=th.float32)
        
        self.save_rgb = save_rgb
        self.rgb_save_dir = rgb_save_dir
        self.episode_count = 0
        self.step_count = 0
        # 上一时刻 semantic（N x 1 x H x W），仅用于 mask 时序特征，不进入 policy 图像输入
        self.prev_semantic = None
        # 上一时刻像素中心 (u, v)，用于构造 du/dv
        self.prev_pixel_uv = None
        
        # New: segmentation map blackout related parameters
        self.blackout_height = blackout_height  # Height at which segmentation map starts to blackout
        self.blackout_transition = blackout_transition  # Transition range
        self.min_blackout_height = blackout_height - blackout_transition  # Height at which segmentation map is fully blacked out
        
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
                    translation = [-translation[2],-translation[0],translation[1]+0.2]  # plus height of the platform
                    break  
            if translation is None:
                translation = [2.0, 0.0, 0.2]  # default
                print(f"Warning: No translation found in {path}, using default: {translation}")
            self.target[i*self.num_agent_per_scene:(i+1)*self.num_agent_per_scene] = \
                th.tensor(translation, device=self.device).float().unsqueeze(0).repeat(self.num_agent_per_scene, 1)
        
        self.step_count = 0
        self.episode_count += 1
        self.prev_semantic = None
        self.prev_pixel_uv = None
        return obs
    
    # def _apply_blackout_effect(self, semantic_obs: np.ndarray) -> np.ndarray:
    #     """
    #     根据当前高度应用分割图变黑效果
    #     当无人机高度低于blackout_height时，分割图逐渐变黑
    #     当高度低于min_blackout_height时，分割图完全变黑
    #     """
    #     if self.position is None or len(self.position) == 0:
    #         return semantic_obs
            
    #     current_heights = self.position[:, 2].detach().numpy() 
    #     result_obs = np.zeros_like(semantic_obs)
        
    #     # 对每个环境单独处理
    #     for env_idx in range(len(current_heights)): 
    #         current_height = current_heights[env_idx]
            
    #         if current_height > self.blackout_height:
    #             result_obs[env_idx] = semantic_obs[env_idx]
    #             continue
                
    #         if current_height <= self.min_blackout_height:
    #             result_obs[env_idx] = np.zeros_like(semantic_obs[env_idx])
    #             continue
                
    #         transition_ratio = (current_height - self.min_blackout_height) / self.blackout_transition
    #         black_image = np.zeros_like(semantic_obs[env_idx])
            
    #         mixed_image = semantic_obs[env_idx] * transition_ratio + black_image * (1 - transition_ratio)
    #         result_obs[env_idx] = mixed_image.astype(semantic_obs[env_idx].dtype)
        
    #     return result_obs
        
    def _world_to_body_velocity(self):
            """
            将世界坐标系下的线性速度转换为机体坐标系下的速度。
            包含 BPTT 安全性修复：Clone 和 Normalization。
            """
            # 1. [关键修复] 使用 .clone() 避免 In-place 操作报错
            # 这样即使物理引擎在下一步修改了 velocity，这里的计算图使用的是这一刻的副本
            vel = self.velocity.clone()
            ori = self.orientation.clone()

            # 2. [关键修复] 归一化四元数，防止梯度爆炸/NaN
            # 添加 eps=1e-8 防止除以零
            norm = ori.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            q = ori / norm

            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            vx_w, vy_w, vz_w = vel[:, 0], vel[:, 1], vel[:, 2]

            # 3. 计算旋转矩阵的逆 (World -> Body)
            # R_inv = R_transpose. 
            # 下面的公式是四元数旋转公式 q* . v . q 的展开形式 (或者 R^T * v)

            # row 1 (Body X axis projection)
            bx = (1 - 2*y**2 - 2*z**2) * vx_w + (2*x*y + 2*w*z) * vy_w + (2*x*z - 2*w*y) * vz_w

            # row 2 (Body Y axis projection)
            by = (2*x*y - 2*w*z) * vx_w + (1 - 2*x**2 - 2*z**2) * vy_w + (2*y*z + 2*w*x) * vz_w

            # row 3 (Body Z axis projection)
            bz = (2*x*z + 2*w*y) * vx_w + (2*y*z - 2*w*x) * vy_w + (1 - 2*x**2 - 2*y**2) * vz_w

            return th.stack([bx, by, bz], dim=1)

    def _world_to_heading_velocity(self):
            """
            将世界坐标系下的线性速度转换为航向系（Heading Frame）。
            只考虑 Yaw 的旋转，忽略 Pitch 和 Roll。
            """
            # 1. 克隆数据，避免 BPTT In-place 错误
            vel = self.velocity.clone()
            q = self.orientation.clone()

            # 2. 从四元数中提取 Yaw (偏航角)
            # 四元数 [w, x, y, z] 转 Yaw 的公式: 
            # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            yaw = th.atan2(siny_cosp, cosy_cosp)

            # 3. 构造旋转矩阵分量
            cos_y = th.cos(yaw)
            sin_y = th.sin(yaw)

            vx_w, vy_w, vz_w = vel[:, 0], vel[:, 1], vel[:, 2]

            # 4. 执行坐标变换 (绕Z轴旋转)
            # Heading_Vx =  cos(yaw) * Vw_x + sin(yaw) * Vw_y
            # Heading_Vy = -sin(yaw) * Vw_x + cos(yaw) * Vw_y
            # Heading_Vz =  Vw_z (高度方向不变)

            vh_x = cos_y * vx_w + sin_y * vy_w
            vh_y = -sin_y * vx_w + cos_y * vy_w
            vh_z = vz_w  # 垂直速度在航向系和世界系是一致的

            return th.stack([vh_x, vh_y, vh_z], dim=1)
    
    def get_observation(
            self,
            indices=None
    ) -> Dict:
        semantic_obs = self.sensor_obs["semantic"].astype(np.float32)

        pixel_obs = []
        for img in semantic_obs:
            nz = np.where(img > 0)
            if len(nz[0]) > 0:
                y_center = np.mean(nz[1])
                x_center = np.mean(nz[2])
                H, W = img.shape[1:3]
                u_norm = (x_center - W/2) / (W/2)
                v_norm = (y_center - H/2) / (H/2)
                visable = 1.0
            else:
                u_norm = 0.0
                v_norm = 0.0
                visable = 0.0
            pixel_obs.append([u_norm, v_norm, visable])
        self.pixel_tensor = th.tensor(pixel_obs, device=self.device, dtype=th.float32)

        if self.prev_pixel_uv is None or self.prev_pixel_uv.shape != self.pixel_tensor[:, :2].shape:
            self.prev_pixel_uv = th.zeros_like(self.pixel_tensor[:, :2])
        d_uv = self.pixel_tensor[:, :2] - self.prev_pixel_uv
        du = d_uv[:, 0:1]
        dv = d_uv[:, 1:2]
        self.prev_pixel_uv = self.pixel_tensor[:, :2].clone()

        # mask 面积差分（与 prev_semantic 对齐；首步 prev 全 0，差分≈当前帧面积）
        if self.prev_semantic is None or self.prev_semantic.shape != semantic_obs.shape:
            self.prev_semantic = np.zeros_like(semantic_obs)
        hn, wn = semantic_obs.shape[2], semantic_obs.shape[3]
        pix = float(hn * wn)
        area_curr = (semantic_obs > 0).astype(np.float32).sum(axis=(1, 2, 3)) / pix
        area_prev = (self.prev_semantic > 0).astype(np.float32).sum(axis=(1, 2, 3)) / pix
        mask_area = th.as_tensor(area_curr, device=self.device, dtype=th.float32).unsqueeze(1)
        d_mask_area = th.as_tensor(area_curr - area_prev, device=self.device, dtype=th.float32).unsqueeze(1)

        self.prev_semantic = semantic_obs.copy()

        # 去掉线速度（body frame linear velocity），保留姿态与角速度；附带视觉时序特征
        state = th.hstack([
            self.pixel_tensor,
            self.orientation,
            self.angular_velocity / 10,
            mask_area,
            d_mask_area,
            du,
            dv,
        ]).to(self.device)
        
        # segmentation map blackout effect (support multi-env)
        # semantic_obs_with_blackout = self._apply_blackout_effect(semantic_obs)

        # if self.save_rgb:
        #     self._save_rgb_images()
        
        self.step_count += 1
        self.total_timesteps += 1

        # if self.total_timesteps % self.target_update_interval == 0:
        #     print("!!!!!!!!!!!! Target update !!!!!!!!!!!!")
        #     self.reset()
            
        # 原代码: 
        # return TensorDict({"state": state, "semantic": th.as_tensor(semantic_obs), })
        
        out = {"state": state}
        if self.semantic_in_obs:
            out["semantic"] = th.as_tensor(semantic_obs)
        return TensorDict(out)

    def get_success(self) -> th.Tensor:
        d_xy = (self.target - self.position)[:, :2].norm(dim=1)
        d_z = (self.target - self.position)[:, 2].abs()
        return (d_xy < self.success_radius) & (d_z < 0.1)
    
    # def get_reward(self) -> Dict[str, th.Tensor]:
        
    #     # 1. x-y plane distance reward
    #     d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
    #     r_xy = -d_xy * 0.04
        
    #     # 2. z axis velocity reward
    #     v_l = (0.5 * self.position[:, 2] - 0).clip(min=0.01, max=0.3)
    #     r_z = 1/(1+(-v_l-self.velocity[:,-1]).abs()) * 0.1
        
    #     # 3. angular velocity penalty
    #     r_omega = (self.angular_velocity - 0).norm(dim=1) * -0.001
        
    #     # 4. orientation penalty
    #     r_ori = (self.orientation - self.target_ori).norm(dim=1) * -0.003

    #     # 5. Yaw angle penalty
    #     w = self.orientation[:, 0]
    #     x = self.orientation[:, 1]
    #     y = self.orientation[:, 2]
    #     z = self.orientation[:, 3]
    #     yaw = th.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    #     r_yaw = yaw.abs() * -0.002 
        
    #     # 6. discrete reward
    #     r_s = 10.0 
    #     r_l = self.get_success() * r_s + self.failure * -1.0
    #     r_smooth = -0.005 * (self._action.clone().cpu() - 0).norm(dim=1) - 0.0025 * (self._action.clone().cpu() - self.last_action).norm(dim=1)
        
    #     # 7. pixel center reward
    #     pixel_dist = (self.pixel_tensor[:,:2]).norm(dim=1)
    #     visible = self.pixel_tensor[:,2]
    #     r_visual = visible * (1.0 - pixel_dist) + (1.0 - visible) * -0.5
    #     r_visual = r_visual * 0.01
        
    #     diff_r = r_xy + r_z + r_omega + r_ori + r_smooth + r_yaw
    #     disc_r = r_l
    #     reward = diff_r + disc_r
        
    #     return {
    #         "diff_r": diff_r, 
    #         "disc_r": disc_r, 
    #         "reward": reward, 
    #         "success": self.success.float(),
    #         "r_z": r_z,
    #         "r_yaw": r_yaw  
    #     }
    
    # with_linear_velocity_28
    def get_reward(self) -> Dict[str, th.Tensor]:
        # 1. x-y plane distance reward
        d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
        r_xy = -d_xy * 0.5
        
        # 2. z axis velocity reward
        v_l = (0.5 * self.position[:, 2] - 0).clip(min=0.01, max=0.3)
        r_z = 1/(1+(-v_l-self.velocity[:,-1]).abs()) * 0.2
        
        # 3. angular velocity penalty
        r_omega = (self.angular_velocity - 0).norm(dim=1) * -0.001
        
        # 4. orientation penalty
        r_ori = (self.orientation - self.target_ori).norm(dim=1) * -0.005

        # 5. Yaw angle penalty
        w = self.orientation[:, 0]
        x = self.orientation[:, 1]
        y = self.orientation[:, 2]
        z = self.orientation[:, 3]
        yaw = th.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        r_yaw = yaw.abs() * -0.002 
        
        # 6. discrete reward (ppo)
        r_s = 10.0 
        r_l = self.get_success() * r_s + self.failure * -1.0
        r_smooth = -0.005 * (self._action.clone().cpu() - 0).norm(dim=1) - 0.0025 * (self._action.clone().cpu() - self.last_action).norm(dim=1)
         
        # 7. pixel center reward
        pixel_dist = (self.pixel_tensor[:,:2]).norm(dim=1)
        visible = self.pixel_tensor[:,2]
        r_visual = visible * (1.0 - pixel_dist) + (1.0 - visible) * -0.5
        r_visual = r_visual * 0.1
        
        diff_r = r_xy + r_z + r_omega + r_smooth + r_visual + r_ori + r_yaw
        # diff_r = r_xy + r_z + r_omega + r_smooth + r_ori
        disc_r = r_l
        reward = diff_r + disc_r
        
        return {
            "diff_r": diff_r, 
            "disc_r": disc_r, 
            "reward": reward, 
            "success": self.success.float(),
            "r_z": r_z,
            "r_xy": r_xy,
            "r_visual": r_visual,
            "r_yaw": r_yaw  
        }
    
    # # with_accelaration_no_linear_velocity_29
    # def get_reward(self) -> Dict[str, th.Tensor]:
    #     # 1. x-y plane distance reward
    #     d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
    #     r_xy = -d_xy * 0.5
        
    #     # 2. z axis velocity reward
    #     v_l = (0.5 * self.position[:, 2] - 0).clip(min=0.01, max=0.3)
    #     r_z = 1/(1+(-v_l-self.velocity[:,-1]).abs()) * 0.2
        
    #     # 3. angular velocity penalty
    #     r_omega = (self.angular_velocity - 0).norm(dim=1) * -0.001
        
    #     # 4. orientation penalty
    #     r_ori = (self.orientation - self.target_ori).norm(dim=1) * -0.005

    #     # # 5. Yaw angle penalty
    #     # w = self.orientation[:, 0]
    #     # x = self.orientation[:, 1]
    #     # y = self.orientation[:, 2]
    #     # z = self.orientation[:, 3]
    #     # yaw = th.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    #     # r_yaw = yaw.abs() * -0.002 
        
    #     # 6. discrete reward (ppo)
    #     r_s = 10.0 
    #     r_l = self.get_success() * r_s + self.failure * -1.0
    #     r_smooth = -0.02 * (self._action.clone().cpu() - 0).norm(dim=1) - 0.0025 * (self._action.clone().cpu() - self.last_action).norm(dim=1)
         
    #     # 7. pixel center reward
    #     pixel_dist = (self.pixel_tensor[:,:2]).norm(dim=1)
    #     visible = self.pixel_tensor[:,2]
    #     r_visual = visible * (1.0 - pixel_dist) + (1.0 - visible) * -0.5
    #     r_visual = r_visual * 0.1    # - sensor_type: SEMANTIC