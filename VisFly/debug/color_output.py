import os
import sys
# from tkinter import Image  # tkinter通常用于GUI，处理图像建议用PIL
from PIL import Image        # 修正为 PIL
import numpy as np
import cv2 as cv
import torch as th
import time

# --- ROS 相关导入 ---
import rospy
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
# ------------------

# from VisFly.utils.common import load_yaml_config

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from VisFly.envs.NavigationEnv import NavigationEnv2
from habitat_sim.sensor import SensorType
from VisFly.utils.maths import Quaternion
from diff_landing.envs.VisualLandingEnv_random_land import VisualLandingEnv

random_kwargs = {
    "state_generator":
        {
            "class": "TargetUniform",
            "kwargs": [
                {"position": {"mean": [3., 0., 4], "half": [0., 0., 0.]}},
            ]
        }
}

scene_path = "VisFly/datasets/visfly-beta/configs/scenes/landing_cube_test"
sensor_kwargs = [{
    "sensor_type": SensorType.COLOR,
    "uuid": "color",
    "resolution": [480, 640],
    "position": [0, 0, -0.05],
}]
scene_kwargs = {
    "path": scene_path,
    "render_settings": {
        "mode": "fix",
        "view": "custom",
        "resolution": [1080, 1920],
        "line_width": 6.,
        "trajectory": True,
        "collision": True
    }
}
num_agent = 1
num_scene = 1 
env = VisualLandingEnv(
    visual=True,
    num_scene=num_scene,
    num_agent_per_scene=num_agent,
    random_kwargs=random_kwargs,
    scene_kwargs=scene_kwargs,
    sensor_kwargs=sensor_kwargs,
    dynamics_kwargs={},
    tensor_output=True,
)

rospy.init_node('visfly_camera_publisher', anonymous=True)
image_pub = rospy.Publisher('/rgb/image_raw', RosImage, queue_size=10)
bridge = CvBridge()
rate = rospy.Rate(30)

print("Start publishing images to ROS topic at 30Hz...")
env.reset()

while not rospy.is_shutdown():
    # 注意：如果不需要每一帧都重置环境（通常只需要step），请根据需求调整 env.reset()
    # 如果是为了仿真飞行，通常是 env.reset() -> while episode: env.step(action)
    # 这里保留你的原始逻辑（每帧reset），但请注意这可能会影响性能
    
    obs = env.get_observation()
    rgb_images = obs["color"]
    
    for i, rgb_image in enumerate(rgb_images):
        
        # 数据处理：VisFly/Habitat通常输出 (C, H, W) 或 (B, C, H, W)，需要转为 numpy
        if isinstance(rgb_image, th.Tensor):
            rgb_image = rgb_image.cpu().numpy()
            
        # 去掉多余维度 (1, C, H, W) -> (C, H, W)
        rgb_image = np.squeeze(rgb_image).astype(np.uint8)
        
        # 调整维度 (C, H, W) -> (H, W, C) 以适配 ROS/OpenCV
        # 假设原始是 (3, 64, 64)，转置后变为 (64, 64, 3)
        rgb_image = np.transpose(rgb_image, (1, 2, 0))
        
        # 确保是有效的 RGB 图像
        if rgb_image.ndim == 3 and rgb_image.shape[2] == 3:
            try:
                # --- 发布图像核心代码 ---
                # 将 Numpy 数组转换为 ROS Image 消息
                # encoding="rgb8" 表示数据顺序是 R-G-B，如果是 BGR 请用 "bgr8"
                ros_msg = bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
                
                # 发布消息
                image_pub.publish(ros_msg)
                # ----------------------
                
            except Exception as e:
                print(f"Error publishing image: {e}")

    # 休眠以维持 30Hz 频率
    rate.sleep()