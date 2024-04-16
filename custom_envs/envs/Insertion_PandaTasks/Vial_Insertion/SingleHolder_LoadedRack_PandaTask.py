import numpy as np

import sys
sys.path.append("..")

from custom_envs.envs.robots.panda import Panda
from custom_envs.envs.Insertion_Tasks.Vial_Insertion.SingleHolder_LoadedRack_Task import SingleHolder_LoadedRack

from panda_chemistry_gym.core import RobotTaskEnv
from panda_chemistry_gym.pybullet import PyBullet


class VialInsertion_SingleHolder_LoadedRack_Env(RobotTaskEnv):

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = SingleHolder_LoadedRack(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position)
        super().__init__(robot, task)
