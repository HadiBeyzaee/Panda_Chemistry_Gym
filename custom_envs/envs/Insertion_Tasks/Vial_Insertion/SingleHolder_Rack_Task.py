
from typing import Any, Dict, Union

import numpy as np

from panda_chemistry_gym.core import Task
from panda_chemistry_gym.pybullet import PyBullet
from panda_chemistry_gym.utils import distance
import os
import math
import random
import pybullet as p

class SingleHolder_Rack(Task):
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
        self.get_ee_position = get_ee_position
        self.Insert_Flag = False


        self.vial_initial_pose =  np.array([0.05, 0.25, 0.005])
        self.vial_initial_pose_holder = np.array([0.05, 0.25, 0.0])


        self.rack_goal =  np.array([0.035 , -0.2 , 0.008])
        self.vial_goal_positions = [
            self.rack_goal + np.array([0.065, 0.0, 0.003]),
            self.rack_goal + np.array([0.12, -0.03, 0.003]),
            self.rack_goal + np.array([0.12, 0.03, 0.003]),
            self.rack_goal + np.array([-0.065, 0.0, 0.003]),
            self.rack_goal + np.array([-0.12, -0.03, 0.003]),
            self.rack_goal + np.array([-0.12, 0.03, 0.003])
        ]
        self.rack_holder =  np.array([0.035, -0.2, 0.0])

        self.ee_z_position = np.array([0.0, 0.0, 0.075])

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.5, width=1.2, height=0.4, x_offset=-0.3)

        # -------------------------------- Vial Initial-----------------------------------------------

        cup_visual = os.getcwd()+'/Objects/Vial.obj'
        cup_collision = os.getcwd()+'/Objects/Vial_vhacd.obj'
        mesh_scale = [0.03] * 3
        visual_kwargs = {
        'fileName': cup_visual,
        'meshScale':mesh_scale,
        'rgbaColor':[1, 0, 0, 1.0]}
       
        collision_kwargs={
        'fileName': cup_collision,
        'meshScale':mesh_scale
        }

        self.sim._create_geometry(
            'vial_initial',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass= 0.1,
            position=self.vial_initial_pose,
            ghost=False,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
        # -------------------------------- Vial Target Position -----------------------------------------------

        cup_visual = os.getcwd()+'/Objects/Vial.obj'
        cup_collision = os.getcwd()+'/Objects/Vial_vhacd.obj'
        mesh_scale = [0.03] * 3
        visual_kwargs = {
        'fileName': cup_visual,
        'meshScale':mesh_scale,
        'rgbaColor': [0, 0, 1, 0.2]}
       
        collision_kwargs={
        'fileName': cup_collision,
        'meshScale':mesh_scale
        }

        self.sim._create_geometry(
            'vial_to_place',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass= 0.0,
            position=self.vial_goal_positions[0],
            ghost=True,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

        #--------------------------------------------------- vial Holder ---------------------
        cup_visual1 = os.getcwd()+'/Objects/single_holder.obj'
        cup_collision1 = os.getcwd()+'/Objects/single_holder_vhacd.obj'      
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
            'single_holder',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1000.0,
            position=self.vial_initial_pose_holder,
            ghost=False,
            visual_kwargs=visual_kwargs1,
            collision_kwargs=collision_kwargs1,
        )

        #------------------------------------- Rack Target Position-----------------------------------------------
        cup_visual1 = os.getcwd()+'/Objects/Rack.obj'
        cup_collision1 = os.getcwd()+'/Objects/Rack_vhacd.obj'

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
            'rack_target',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1.0,
            position=self.rack_goal,
            ghost=False,
            visual_kwargs=visual_kwargs1,
            collision_kwargs=collision_kwargs1,
        )

        #--------------------------------------------Rack holder Target ------------------------------
        cup_visual1 = os.getcwd()+'/Objects/Rack_holder.obj'
        cup_collision1 = os.getcwd()+'/Objects/Rack_holder_vhacd.obj'

        mesh_scale = [0.03] * 3
        visual_kwargs1 = {
        'fileName': cup_visual1,
        'meshScale':mesh_scale,
        'rgbaColor': [0.0, 0.0, 0.0, 1.0]}
       
        collision_kwargs1={
        'fileName': cup_collision1,
        'meshScale':mesh_scale
        }

        self.sim._create_geometry(
            'rack_holder_target',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1000.0,
            position=self.rack_holder,
            ghost=False,
            visual_kwargs=visual_kwargs1,
            collision_kwargs=collision_kwargs1,
        )

    # ------------------------------------ Observation space for the objects ---------------------------------
    def get_obs(self) -> np.ndarray:
        object_position = self.sim.get_base_position("vial_initial")
        object_rotation = self.sim.get_base_rotation("vial_initial")
        object_velocity = self.sim.get_base_velocity("vial_initial")

        observation = np.concatenate([object_position, object_rotation, object_velocity])
        return observation
       
    # ---------------------------------- Achieved Goal ----------------------------------------------------------------
    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("vial_initial"))
        ee_position = self.get_ee_position()

        return np.concatenate((object_position , ee_position))
    

    # --------------------------------- Reset the Environment ----------------------------------------------------------  
    def reset(self) -> None:

        self.q = [0,0,0,0]
        r = math.pi
        self.q[0] = math.cos(r/4)
        self.q[1] = math.sin(r/4)
        self.q[2] = math.sin(r/4)
        self.q[3] = math.cos(r/4) 

        self.Insert_Flag = False
        rand1 = np.array([random.random()*0.04, random.random()*0.04, 0])
        rand2 = np.array([random.random()*0.04, random.random()*0.04, 0])


        self.goal1, self.targrand = self._sample_goal() 
        self.goal1 = self.goal1 + rand2                
        self.goal2 = self.goal1 + self.ee_z_position
        self.goal = np.concatenate([self.goal1, self.goal2])

        self.sim.set_base_pose("vial_initial", self.vial_initial_pose + rand1, np.array(self.q))
        self.sim.set_base_pose("single_holder",  self.vial_initial_pose_holder + rand1, np.array(self.q))

        self.sim.set_base_pose("rack_target",  self.rack_goal + rand2, np.array(self.q))
        self.sim.set_base_pose("rack_holder_target",  self.rack_holder + rand2, np.array(self.q))

        self.sim.set_base_pose("vial_to_place",  self.goal1, np.array(self.q))
        

    # ---------- Select a random well between six wells of the rack for the target position of the vial ------------
    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        rand = random.randint(1,6)
        if rand == 1:
            goal = self.vial_goal_positions[0]
        elif rand == 2:
            goal = self.vial_goal_positions[1]
        elif rand == 3:
            goal = self.vial_goal_positions[2]
        elif rand == 4:
            goal = self.vial_goal_positions[3]
        elif rand == 5:
            goal = self.vial_goal_positions[4]
        else:
            goal = self.vial_goal_positions[5]

        return goal, rand
    
    # -------------------------------------- Success Flag -----------------------------------------------------------------
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        self.Insert_Flag = np.array(d < self.distance_threshold, dtype=bool)

        return np.array((d ) < self.distance_threshold, dtype=bool)


    # ------------------------------------ Compute the Reward -------------------------------------------------------------
    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:

        contact_point1 = p.getContactPoints(
        bodyA=self.sim._bodies_idx['vial_initial'],
        bodyB=self.sim._bodies_idx['single_holder'])

        d = distance(achieved_goal, desired_goal)

        if len(contact_point1) != 0:
            reward = -1.2
        else:
            reward = -d

            if self.Insert_Flag == True:
                    reward = 2.0

        return reward
