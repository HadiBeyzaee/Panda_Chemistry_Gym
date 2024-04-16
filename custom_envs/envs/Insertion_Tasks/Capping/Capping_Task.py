
from typing import Any, Dict, Union

import numpy as np

from panda_chemistry_gym.core import Task
from panda_chemistry_gym.pybullet import PyBullet
from panda_chemistry_gym.utils import distance
import os
import math
import random
import pybullet as p


class Capping(Task):
    def __init__(
            self,
            sim: PyBullet,
            get_ee_position ,
            reward_type: str = "sparse",
            distance_threshold: float = 0.015
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.Insert_Flag = False
        self.get_ee_position = get_ee_position

        self.cap_position =  np.array([0.03, 0.2, 0.063])
        self.cap_holder_position = np.array([0.06, 0.2, 0.0])

        self.vial_position = np.array([0.06, -0.2, 0.08])
        self.vial_holder_position = np.array([0.06, -0.2, 0.0])

        self.vial_ee = np.array([0.06, -0.2, 0.148])
        self.cap_target_position = np.array([0.06, -0.2, 0.147])
        

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.5, width=1.0, height=0.4, x_offset=-0.3)

        # ---------------------------------- Cap -----------------------------------------------

        cup_visual = os.getcwd()+'/Objects/cap.obj'
        cup_collision = os.getcwd()+'/Objects/cap_vhacd.obj'
        mesh_scale = [0.03] * 3
        visual_kwargs = {
        'fileName': cup_visual,
        'meshScale':mesh_scale,
        'rgbaColor':[0, 0, 1, 1.0]}
       
        collision_kwargs={
        'fileName': cup_collision,
        'meshScale':mesh_scale
        }

        self.sim._create_geometry(
            'cap',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=  0.05,
            position=self.cap_position,
            ghost=False,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
        #--------------------------------------------------- Cap Holder ---------------------
        cup_visual1 = os.getcwd()+'/Objects/cap_holder.obj'
        cup_collision1 = os.getcwd()+'/Objects/cap_holder_vhacd.obj'       
        mesh_scale = [0.03] * 3
        visual_kwargs1 = {
        'fileName': cup_visual1,
        'meshScale':mesh_scale,
        'rgbaColor':[0.5, 0.5, 0.5, 1.0]}
       
        collision_kwargs1={
        'fileName': cup_collision1,
        'meshScale':mesh_scale
        }

        self.sim._create_geometry(
            'cap_holder',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1000.0,
            position=self.cap_holder_position,
            ghost=False,
            visual_kwargs=visual_kwargs1,
            collision_kwargs=collision_kwargs1,
        )

        # -------------------------------------- Vial for Capping -----------------------------------------------

        cup_visual = os.getcwd()+'/Objects/Vial_uncapped.obj'
        cup_collision = os.getcwd()+'/Objects/Vial_uncapped_vhacd.obj'
        mesh_scale = [0.03] * 3
        visual_kwargs = {
        'fileName': cup_visual,
        'meshScale':mesh_scale,
        'rgbaColor':[1, 0, 0, 1.0]
        }
       
        collision_kwargs={
        'fileName': cup_collision,
        'meshScale':mesh_scale
        }

        self.sim._create_geometry(
            'vial',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass= 0.1,
            position=self.vial_position,
            ghost=False,
            visual_kwargs=visual_kwargs,
            lateral_friction= 1.0,
            collision_kwargs=collision_kwargs
        )
        visual_kwargs1 = {
        'fileName': cup_visual,
        'meshScale':mesh_scale,
        'rgbaColor':[0, 1, 0, 0.1]
        }

        #--------------------------------------------------- Vial Holder ----------------------------------
        cup_visual1 = os.getcwd()+'/Objects/Vial_holder_capping.obj'
        cup_collision1 = os.getcwd()+'/Objects/Vial_holder_capping_vhacd.obj'   

        mesh_scale = [0.03] * 3
        visual_kwargs1 = {
        'fileName': cup_visual1,
        'meshScale':mesh_scale,
        'rgbaColor':[0.5, 0.5, 0.5, 1.0]}
       
        collision_kwargs1={
        'fileName': cup_collision1,
        'meshScale':mesh_scale
        }

        self.sim._create_geometry(
            'vial_holder',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1000.0,
            position=self.vial_holder_position,
            ghost=False,
            visual_kwargs=visual_kwargs1,
            collision_kwargs=collision_kwargs1,
        )
    # ------------------------------------ Observation space for the objects ---------------------------------
    def get_obs(self) -> np.ndarray:
        object_position = self.sim.get_base_position("cap")
        object_rotation = self.sim.get_base_rotation("cap")
        object_velocity = self.sim.get_base_velocity("cap")

        observation = np.concatenate([object_position, object_rotation, object_velocity])
        return observation
       
    # ---------------------------------- Achieved Goal ---------------------------------------------------------
    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("cap"))
        ee_position = self.get_ee_position()

        return np.concatenate((object_position , ee_position))

    # --------------------------------- Reset the Environment----------------------------------------------------
    def reset(self) -> None:
        self.Insert_Flag = False

        rand1 = np.array([random.random()*0.05, random.random()*0.05 , 0])
        rand2 = np.array([random.random()*0.05, random.random()*0.05 , 0])

        self.goal = self._sample_goal() + np.concatenate([rand2 , rand2])

        self.q = [0, 0, 0, 0]
        r = math.pi
        self.q[0] = math.cos(r/4)
        self.q[1] = math.sin(r/4)
        self.q[2] = math.sin(r/4)
        self.q[3] = math.cos(r/4)

        self.sim.set_base_pose("cap_holder",  self.cap_holder_position + rand1 , self.q)
        self.sim.set_base_pose("cap", self.cap_position + rand1 , self.q)

        self.sim.set_base_pose("vial_holder", self.vial_holder_position + rand2, self.q)
        self.sim.set_base_pose("vial", self.vial_position + rand2 , self.q)


    def _sample_goal(self) -> np.ndarray:
        goal1 = self.cap_target_position
        goal2 = self.vial_ee

        return np.concatenate([goal1 , goal2])

    # ---------------------- Success Flag -----------------------------------------------
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        self.Insert_Flag = np.array(d < self.distance_threshold, dtype=bool)
        return np.array(d < self.distance_threshold, dtype=bool)

    # ------------------------------------ Compute the Reward -------------------------------------------------------------
    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:

        contact_point = p.getContactPoints(
            bodyA=self.sim._bodies_idx['cap'],
            bodyB=self.sim._bodies_idx['cap_holder'],
        )

        d = distance(achieved_goal, desired_goal)


        if len(contact_point) != 0:
            reward = -1.2
        else:
                
            reward = -d
            if self.Insert_Flag == True:
                reward = 1.0

        return reward

