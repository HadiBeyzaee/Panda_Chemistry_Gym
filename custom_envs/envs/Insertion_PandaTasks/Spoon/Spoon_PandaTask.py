import numpy as np

import sys
sys.path.append("..")

from custom_envs.envs.robots.panda import Panda
from custom_envs.envs.Insertion_Tasks.Spoon.Spoon_Task import Spoon

from panda_chemistry_gym.core import RobotTaskEnv
from panda_chemistry_gym.pybullet import PyBullet


class Spoon_Env(RobotTaskEnv):

    def __init__(self, render: bool = False, reward_type: str = "dense", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Spoon(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position)
        super().__init__(robot, task)
