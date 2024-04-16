
from typing import Any, Dict, Union

import numpy as np

from panda_chemistry_gym.core import Task
from panda_chemistry_gym.pybullet import PyBullet
from panda_chemistry_gym.utils import distance
import os
import math
import random
import pybullet as p

class Loaded_Rack(Task):
   
    def __init__(
            self,
            sim: PyBullet,
            get_ee_position ,
            reward_type: str = "sparse",
            distance_threshold: float = 0.02
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.Insert_Flag = False
        self.get_ee_position = get_ee_position

        self.rack_initial =  np.array([0.05, 0.2, 0.008])
        self.vial_initial1 = self.rack_initial + np.array([0.065, 0.0, 0.005])  
        self.vial_initial2 = self.rack_initial + np.array([0.12, -0.03, 0.005])  
        self.vial_initial3 = self.rack_initial + np.array([0.12, 0.03, 0.005])  
        self.vial_initial4 = self.rack_initial + np.array([-0.065, 0.0, 0.005])  
        self.vial_initial5 = self.rack_initial + np.array([-0.12, -0.03, 0.005])  
        self.vial_initial6 = self.rack_initial + np.array([-0.12, 0.03, 0.005])  
        self.rack_holder1 =  np.array([0.05, 0.2, 0.0])

        self.rack_goal =  np.array([0.05 , -0.25 , 0.008])
        self.rack_holder2 =  np.array([0.05 , -0.25 , 0.0])

        self.ee_target =  np.array([0.05 , -0.25 , 0.075])


        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.5, width=1.0, height=0.4, x_offset=-0.3)

        # -------------------------------- Vial Initial Position -----------------------------------------------

        cup_visual = os.getcwd()+'/Objects/Vial.obj'
        cup_collision = os.getcwd()+'/Objects/Vial_vhacd.obj'
        mesh_scale = [0.03] * 3
        visual_kwargs = {
        'fileName': cup_visual,
        'meshScale':mesh_scale,
        'rgbaColor':[1, 1, 1, 1]}
       
        collision_kwargs={
        'fileName': cup_collision,
        'meshScale':mesh_scale
        }

        self.sim._create_geometry(
            'vial1',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass= 0.1,
            position=self.vial_initial1,
            ghost=False,
            lateral_friction = 0.5,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
        self.sim._create_geometry(
            'vial2',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass= 0.1,
            position=self.vial_initial2,
            ghost=False,
            lateral_friction = 0.5,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
        self.sim._create_geometry(
            'vial3',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass= 0.1,
            position=self.vial_initial3,
            ghost=False,
            lateral_friction = 0.5,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
        self.sim._create_geometry(
            'vial4',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass= 0.1,
            position=self.vial_initial4,
            ghost=False,
            lateral_friction = 0.5,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
        self.sim._create_geometry(
            'vial5',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass= 0.1,
            position=self.vial_initial5,
            ghost=False,
            lateral_friction = 0.5,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )  
        self.sim._create_geometry(
            'vial6',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass= 0.1,
            position=self.vial_initial6,
            ghost=False,
            lateral_friction = 0.5,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )  

        #------------------------------------- Rack Initial Position --------------------------------------
        cup_visual1 = os.getcwd()+'/Objects/Rack.obj'
        cup_collision1 = os.getcwd()+'/Objects/Rack_vhacd2.obj'
        mesh_scale = [0.03] * 3
        visual_kwargs1 = {
        'fileName': cup_visual1,
        'meshScale':mesh_scale,
        'rgbaColor':[0.5, 0.5, 0.5, 1]}
       
        collision_kwargs1={
        'fileName': cup_collision1,
        'meshScale':mesh_scale
        }


        self.sim._create_geometry(
            'rack_initial',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=0.6,
            position=self.rack_initial,
            ghost=False,
            visual_kwargs=visual_kwargs1,
            collision_kwargs=collision_kwargs1,
        )

        #------------------------------------- Rack holder -----------------------------------------------
        cup_visual1 = os.getcwd()+'/Objects/Rack_holder.obj'
        cup_collision1 = os.getcwd()+'/Objects/Rack_holder_vhacd.obj'
        mesh_scale = [0.03] * 3
        visual_kwargs1 = {
        'fileName': cup_visual1,
        'meshScale':mesh_scale,
        'rgbaColor': [0.0, 0.0, 0.0, 1]}
       
        collision_kwargs1={
        'fileName': cup_collision1,
        'meshScale':mesh_scale
        }

        self.sim._create_geometry(
            'rack_holder_initial',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1000.0,
            position=self.rack_holder1,
            ghost=False,
            visual_kwargs=visual_kwargs1,
            collision_kwargs=collision_kwargs1,
        )
        self.sim._create_geometry(
            'rack_holder_target',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1000.0,
            position=self.rack_holder2,
            ghost=False,
            visual_kwargs=visual_kwargs1,
            collision_kwargs=collision_kwargs1,
        )
    # ------------------------------------ Observation space for the objects ---------------------------------
    def get_obs(self) -> np.ndarray:
        object_position = self.sim.get_base_position("rack_initial")
        object_rotation = self.sim.get_base_rotation("rack_initial")
        object_velocity = self.sim.get_base_velocity("rack_initial")

        observation = np.concatenate([object_position, object_rotation, object_velocity])
        return observation
       
    # ---------------------------------- Achieved Goal --------------------------------------------------
    def get_achieved_goal(self) -> np.ndarray:
        object_position = self.sim.get_base_position("rack_initial") 
        ee_position = np.array(self.get_ee_position())
        achieved =  np.concatenate([object_position, ee_position])
       
        return achieved

    # --------------------------------- Reset -----------------------------------------------------------
    def reset(self) -> None:

        self.q = [0,0,0,0]
        r = math.pi
        self.q[0] = math.cos(r/4)
        self.q[1] = math.sin(r/4)
        self.q[2] = math.sin(r/4)
        self.q[3] = math.cos(r/4) 

        rand1 = np.array([random.random()*0.05, random.random()*0.05, 0])
        rand2 = np.array([random.random()*0.05, random.random()*0.05, 0])

        self.Insert_Flag = False
        rack_target = self.rack_goal + rand2
        ee_target = self.ee_target + rand2
        self.goal = np.concatenate([rack_target, ee_target]) 

        self.sim.set_base_pose("vial1", self.vial_initial1 + rand1, np.array(self.q))
        self.sim.set_base_pose("vial2", self.vial_initial2 + rand1, np.array(self.q))
        self.sim.set_base_pose("vial3", self.vial_initial3 + rand1, np.array(self.q))
        self.sim.set_base_pose("vial4", self.vial_initial4 + rand1, np.array(self.q))
        self.sim.set_base_pose("vial5", self.vial_initial5 + rand1, np.array(self.q))
        self.sim.set_base_pose("vial6", self.vial_initial6 + rand1, np.array(self.q))
        self.sim.set_base_pose("rack_initial",  self.rack_initial + rand1, np.array(self.q))
        self.sim.set_base_pose("rack_holder_initial",  self.rack_holder1 + rand1, np.array(self.q))

        self.sim.set_base_pose("rack_holder_target",  self.rack_holder2 + rand2, np.array(self.q))
       

    # ------------------------------------ Success Flag -----------------------------------------------------------------
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:

        vial_bodies = ['vial1', 'vial2', 'vial3', 'vial4', 'vial5', 'vial6']
        
        vial_contact_counts = [len(p.getContactPoints(bodyA=self.sim._bodies_idx[vial], bodyB=self.sim._bodies_idx['rack_initial'],)) for vial in vial_bodies]

        d = distance(achieved_goal, desired_goal)
        self.Insert_Flag = np.array(d < self.distance_threshold, dtype=bool)

        if any(vial_contact_count == 0 for vial_contact_count in vial_contact_counts):
            penalty = 0.1
        else:
            penalty = 0.0

        return np.array((d + penalty) < self.distance_threshold, dtype=bool)
    

    # ------------------------------------ Compute the Reward -------------------------------------------------------------
    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        contact_point = p.getContactPoints(
            bodyA=self.sim._bodies_idx['rack_initial'],
            bodyB=self.sim._bodies_idx['rack_holder_initial'],
        )
        vial_bodies = ['vial1', 'vial2', 'vial3', 'vial4', 'vial5', 'vial6']
        
        vial_contact_counts = [len(p.getContactPoints(bodyA=self.sim._bodies_idx[vial], bodyB=self.sim._bodies_idx['rack_initial'],
                                                                                        )) for vial in vial_bodies]
        
        d = distance(achieved_goal, desired_goal)

        if len(contact_point) != 0:
            reward = -1.2
        else:
            reward = -d
            if any(vial_contact_count == 0 for vial_contact_count in vial_contact_counts):
                reward = reward - 0.5

            if self.Insert_Flag == True:
                 if all(vial_contact_count != 0 for vial_contact_count in vial_contact_counts):
                     reward = 5.0
                 else:
                     reward = -1.0

        return reward

