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
    ):
        sensor_kwargs = [{
            "sensor_type": SensorType.SEMANTIC,
            "uuid": "semantic",
            "resolution": [64, 64],
            "position": [0, 0, -0.1],
            "orientation": [-np.pi / 2, 0, 0]
        }]
        random_kwargs = {
            "state_generator":
                {
                    "class": "Uniform",
                    "kwargs": [
                        {"position": {"mean": [3., 0., 3.0], "half": [1.0, 1.0, 0.5]}},
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
        # self.observation_space["state"] = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        self.observation_space["state"] = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        # self.observation_space["color"] = spaces.Box(low=0, high=255, shape=(1,64,64), dtype=np.float32) # (C, H, W)
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
        self.success_radius = 0.2
        # self.envs.sceneManager.scene

    def get_observation(
            self,
            indices=None
    ) -> Dict:
        # self.previous_position.append(self.position.clone())
        # self.previous_actions.append(self._action.clone())
        
        # if len(self.previous_position) > 1:
        #     self.last_position= self.previous_position[-2]
        # if len(self.previous_actions) > 2:
        #     self.pastAction = th.cat(list(self.previous_actions)[:3], dim=-1)
        #     self.last_action = self.previous_actions[-2] #倒数第二个应该才是上一步的动作
        
        # semantic_img = self.sensor_obs["semantic"][0]  # 取第一个环境的图像
        # vis_img = semantic_img[0]  # 取第一个通道 (形状变为 [64, 64])

        # # 归一化到0-255范围以便显示
        # if vis_img.max() > 0:
        #     vis_img = (vis_img / vis_img.max() * 255).astype(np.uint8)
        # else:
        #     vis_img = vis_img.astype(np.uint8)
            
        # cv2.imshow("Semantic", vis_img)
        # cv2.waitKey(1)
        
        # state = th.hstack([
        #     (self.target - self.position) / self.max_sense_radius,
        #     self.orientation,
        #     self.velocity / 10,
        #     self.angular_velocity / 10,
        # ]).to(self.device)
        
        state = th.hstack([
            self.position / self.max_sense_radius,
            self.orientation,
            self.velocity / 10,
            self.angular_velocity / 10,
        ]).to(self.device)
        
        # bgr_image = th.as_tensor(self.sensor_obs["color"], device=self.device, dtype=th.float32)
        # R = bgr_image[:, 0, :, :]  
        # G = bgr_image[:, 1, :, :]  
        # B = bgr_image[:, 2, :, :]  
        # gray_image = 0.299 * R + 0.587 * G + 0.114 * B
        # gray_image = gray_image.unsqueeze(1)
        # gray_image = gray_image / 255.0
        
        # debug
        # img_to_show = gray_image[0].squeeze(0).detach().cpu().numpy()  # 形状 (64, 48)
        # img_to_show = (img_to_show * 255).astype(np.uint8)  # 转换为uint8
        # cv2.imshow("gray", img_to_show)
        # cv2.waitKey(1)  # 使用1ms而不是0，避免阻塞
        
        return TensorDict({
            "state": state,
            # "color": gray_image,
            "semantic": th.as_tensor(self.sensor_obs["semantic"].astype(np.float32))
            # "depth": gray_image
            # "color": th.as_tensor(self.sensor_obs["color"], dtype=th.float32)
        })

    # def get_success(self) -> th.Tensor:
    #     landing_half = 0.3
    #     # return th.full((self.num_envs,), False)
    #     return (self.position[:, 2] <= 0.2) \
    #         & (self.position[:, :2] < (self.target[:2] + landing_half)).all(dim=1)\
    #            & (self.position[:, :2] > (self.target[:2] - landing_half)).all(dim=1) \
    #            & (self.velocity.norm(dim=1) <= 0.3)  #
    #     # & \
    #     # ((self.position[:, :2] < self.target[:2] + landing_half).all(dim=1) & (self.position[:, :2] > self.target[:2] - landing_half).all(dim=1))

    def get_success(self) -> th.Tensor:
        # return th.full((self.num_agent,), False)
        return (self.position - self.target).norm(dim=1) < self.success_radius
    
    # def get_reward(self) -> Dict[str, th.Tensor]:
    #     lambda_xy, lambda_z, lambda_center = 0.4, 1.0, 0.6
    #     zeta, rho, d_s = 30.0, 20.0, 10.0
    #     safe_vel = -0.5
        
    #     visibility, pad_centers = self.detect_landing_pad()
        
    #     img_center = th.tensor([32, 32], device=self.device, dtype=th.float32)
    #     center_dist = th.norm(pad_centers - img_center, dim=1)
    #     max_dist = th.norm(th.tensor([64, 64], device=self.device).float())
    #     norm_dist = center_dist / max_dist
    #     R_center = visibility * th.exp(-5.0 * norm_dist)
        
    #     xy_dist = (self.position[:, :2] - self.target[:, :2]).norm(dim=1)
    #     R_xy = th.zeros_like(xy_dist)

    #     visible = visibility > 0.5
    #     if visible.any():
    #         vis_mask = visible.bool()
    #         R_xy[vis_mask] = (rho**((d_s - xy_dist[vis_mask])/d_s) - 1) / (rho - 1)

    #     v_z = self.velocity[:, 2] 
    #     R_z = th.zeros_like(v_z)
        
    #     safe = (v_z <= 0) & (v_z >= safe_vel)
    #     unsafe = ~safe

    #     R_z[safe] = (zeta**(v_z[safe]/safe_vel) - 1) / (zeta - 1)
    #     R_z[unsafe] = -0.1 * th.abs(v_z[unsafe] - safe_vel)

    #     reward = lambda_xy * R_xy + lambda_z * R_z + lambda_center * R_center
        
    #     return {
    #         "reward": reward,
    #         "R_xy": R_xy,
    #         "R_z": R_z,
    #         "R_center": R_center,
    #         "visibility": visibility
    #     }
    
    def get_reward(self) -> th.Tensor:
        # eta = th.as_tensor(1.2)
        v_l = (0.5 * self.position[:, 2] - 0).clip(min=0.05, max=0.5).clone().detach()
        # v_l = 1 * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
        # r_p = -0.1
        # # base_r = th.ones((self.num_envs,)) * 0.0
        # descent_v = -self.velocity[:, 2] - 0
        # r_z_punish = ((descent_v > v_l) | (descent_v < 0))
        # r_z = r_z_punish * r_p + \
        #       ~r_z_punish * (eta.pow(descent_v / v_l) - 1) / (eta-1) * 0.1

        # r_z_first = descent_v <= v_l
        # r_z = ~r_z_first * (eta.pow(-4 * descent_v / v_l + 5) - 1) / (eta - 1) * 0.1 + \
        #       r_z_first * (eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1
        # r_z_first * (eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1
        # d_z = (self.target - self.position)[:, 2].abs()
        # r_z = -0.02 * d_z
        # r_z = -(-v_l-self.velocity[:,-1]).abs() * 0.02

        r_z = 1/(1+(-v_l-self.velocity[:,-1]).abs()) * 0.1

        # rho = th.as_tensor(1.2)
        # d_s = 2. * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
        d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
        # r_xy_punish = d_xy > d_s
        # r_xy = (rho.pow(1 - d_xy / d_s) - 1) / (rho - 1) * 0.1
        r_xy = -d_xy * 0.02 * 2
        d_x = (self.target - self.position)[:,0].abs()
        d_y = (self.target - self.position)[:,1].abs()
        r_x = -d_x * 0.02
        r_y = -d_y * 0.02
        # r_cmd = -0.001 * (self._action - 0).norm(dim=1) - 0.001 * (self._action - self.last_action).norm(dim=1)
        r_omega = (self.angular_velocity - 0).norm(dim=1) * -0.001
        r_ori = (self.orientation - self.target_ori).norm(dim=1) * -0.003
        r_s = 10.
        # reward = r_l + r_xy + r_z + r_omega #+ base_r
        if self.r_type == 0:
            r_s = 0
        elif self.r_type == 1:
            r_s = 5
        elif self.r_type == 2:
            r_s = 10
        elif self.r_type == 3:
            r_s = 15
        elif self.r_type == 4:
            r_s = 20
        else:
            raise ValueError(f"Invalid reward type: {self.r_type}")
        landing_error = (self.position - self.target).norm(dim=1)
        r_l = self.success * r_s + self.failure * -0.1
        diff_r = r_xy + r_z + r_omega + r_ori
        disc_r = r_l
        reward = diff_r + disc_r

        return {"diff_r":diff_r, "disc_r":disc_r, "reward":reward, "landing_error":landing_error/100,"success":self.success.float()}

    # def detect_landing_pad(self) -> Tuple[th.Tensor, th.Tensor]:
    #     rgb_images = self.sensor_obs["color"]
    #     num_envs, _,height, width= rgb_images.shape
        
    #     pad_visible = np.zeros(num_envs, dtype=bool)
        
    #     # 根据您的实际标签类型配置，常用选项: tag36h11, tag25h9, tag16h5
    #     detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))
    #     # 添加可见性置信度
    #     visibility_confidence = np.zeros(num_envs, dtype=np.float32)
    #     for i in range(num_envs):
    #         img = rgb_images[i]
            
    #         R = img[0, :, :]  
    #         G = img[1, :, :]  
    #         B = img[2, :, :]  
    #         gray_image = 0.299 * R + 0.587 * G + 0.114 * B
    #         gray_image = gray_image.astype(np.uint8)  # 转换为uint8
    #         tags = detector.detect(gray_image)

    #         valid_tag_found = False
    #         for tag in tags:
    #             # 检查标签置信度
    #             if tag.decision_margin < 10:  # 置信度过低，可能是误检
    #                 continue
                    
    #             # 检查标签大小 (避免远距离小标签)
    #             tag_size = max(tag.corners[:, 0].max() - tag.corners[:, 0].min(),
    #                         tag.corners[:, 1].max() - tag.corners[:, 1].min())
    #             if tag_size < 0.05 * min(height, width):  # 标签太小，忽略
    #                 continue
                    
    #             # 检查标签是否在图像中心区域 (可选)
    #             center_x, center_y = np.mean(tag.corners, axis=0)
    #             if (center_x < width * 0.2 or center_x > width * 0.8 or 
    #                 center_y < height * 0.2 or center_y > height * 0.8):
    #                 # 标签在边缘区域，可能是部分可见
    #                 continue
                    
    #             # 如果配置了特定ID，可以在这里过滤
    #             # if tag.tag_id == self.target_tag_id:
    #             valid_tag_found = True
    #             break
            
    #         pad_visible[i] = valid_tag_found
        
    #     return th.tensor(pad_visible, device=self.device)