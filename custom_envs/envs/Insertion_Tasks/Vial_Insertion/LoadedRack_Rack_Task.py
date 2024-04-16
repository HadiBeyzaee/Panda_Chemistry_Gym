
from typing import Any, Dict, Union

import numpy as np

from panda_chemistry_gym.core import Task
from panda_chemistry_gym.pybullet import PyBullet
from panda_chemistry_gym.utils import distance
import os
import math
import random
import pybullet as p

class LoadedRack_Rack(Task):
    
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


        self.rack_initial =  np.array([0.01, 0.25, 0.008])
        self.vial_initial_positions = [
            self.rack_initial + np.array([0.065, 0.0, 0.01]),
            self.rack_initial + np.array([0.12, -0.03, 0.01]),
            self.rack_initial + np.array([0.12, 0.03, 0.01]),
            self.rack_initial + np.array([-0.065, 0.0, 0.01]),
            self.rack_initial + np.array([-0.12, -0.03, 0.01]),
            self.rack_initial + np.array([-0.12, 0.03, 0.01])
        ]
        self.rack_holder1 =  np.array([0.01, 0.25, 0.0])


        self.rack_goal =  np.array([0.035 , -0.2, 0.008])
        self.vial_goal_positions = [
            self.rack_goal + np.array([0.065, 0.0, 0.003]),
            self.rack_goal + np.array([0.12, -0.03, 0.003]),
            self.rack_goal + np.array([0.12, 0.03, 0.003]),
            self.rack_goal + np.array([-0.065, 0.0, 0.003]),
            self.rack_goal + np.array([-0.12, -0.03, 0.003]),
            self.rack_goal + np.array([-0.12, 0.03, 0.003])
        ]
        self.rack_holder2 =  np.array([0.035 , -0.2 , 0.0])


        self.ee_z_position = np.array([0.0, 0.0, 0.075])

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.8, width=1.2, height=0.4, x_offset=-0.3)

        # -------------------------------- Vial Initial Position -----------------------------------------------

        cup_visual = os.getcwd()+'/Objects/Vial.obj'
        cup_collision = os.getcwd()+'/Objects/Vial_vhacd.obj'
        mesh_scale = [0.03] * 3
        visual_kwargs = {
        'fileName': cup_visual,
        'meshScale':mesh_scale,
        'rgbaColor': [1, 0, 0, 1.0]}
       
        collision_kwargs={
        'fileName': cup_collision,
        'meshScale':mesh_scale
        }

        self.sim._create_geometry(
            'vial_to_pick',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass= 0.1,
            position=self.vial_initial_positions[0],
            ghost=False,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
     # -------------------------------- Vial Initial Position -----------------------------------------------

        cup_visual = os.getcwd()+'/Objects/Vial.obj'
        cup_collision = os.getcwd()+'/Objects/Vial_vhacd.obj'
        mesh_scale = [0.03] * 3
        visual_kwargs = {
        'fileName': cup_visual,
        'meshScale':mesh_scale,
        'rgbaColor': [1, 1, 1, 1]}
       
        collision_kwargs={
        'fileName': cup_collision,
        'meshScale':mesh_scale
        }

        self.sim._create_geometry(
            'vial1',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass= 0.1,
            position=self.vial_initial_positions[1],
            ghost=False,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
        self.sim._create_geometry(
            'vial2',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass= 0.1,
            position=self.vial_initial_positions[2],
            ghost=False,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
        self.sim._create_geometry(
            'vial3',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass= 0.1,
            position=self.vial_initial_positions[3],
            ghost=False,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
        self.sim._create_geometry(
            'vial4',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass= 0.1,
            position=self.vial_initial_positions[4],
            ghost=False,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
        self.sim._create_geometry(
            'vial5',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass= 0.1,
            position=self.vial_initial_positions[5],
            ghost=False,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )  

        # -------------------------------------- Vial Target  ---------------------------------------------
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
            'vial_target',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass= 0.0,
            position=self.vial_goal_positions[0],
            ghost=True,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

        #------------------------------------- Rack Initial Position --------------------------------------
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
            'rack_initial',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1000,
            position=self.rack_initial,
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
        #------------------------------------- Rack holder Initial-----------------------------------------------
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
            'rack_holder_initial',
            geom_type=self.sim.physics_client.GEOM_MESH,
            mass=1000.0,
            position=self.rack_holder1,
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
            position=self.rack_holder2,
            ghost=False,
            visual_kwargs=visual_kwargs1,
            collision_kwargs=collision_kwargs1,
        )
    # ------------------------------------ Observation space for the objects ---------------------------------
    def get_obs(self) -> np.ndarray:
        object_position = self.sim.get_base_position("vial_to_pick")
        object_rotation = self.sim.get_base_rotation("vial_to_pick")
        object_velocity = self.sim.get_base_velocity("vial_to_pick")

        observation = np.concatenate([object_position, object_rotation, object_velocity])
        return observation
       
    # ---------------------------------- Achieved Goal --------------------------------------------------
    def get_achieved_goal(self) -> np.ndarray:

        object_position = np.array(self.sim.get_base_position("vial_to_pick"))
        ee_position = np.array(self.get_ee_position())
        achieved = np.concatenate([ object_position, ee_position])

        return achieved

    # --------------------------------- Reset the Environment-----------------------------------------------------------
    def reset(self) -> None:
        
        self.q = [0,0,0,0]
        r = math.pi
        self.q[0] = math.cos(r/4)
        self.q[1] = math.sin(r/4)
        self.q[2] = math.sin(r/4)
        self.q[3] = math.cos(r/4) 
        
        self.Insert_Flag = False
        rand1 = np.array([random.random()*0.05, random.random()*0.05, 0])
        rand2 = np.array([random.random()*0.05, random.random()*0.05, 0])

        self.object_position, self.objrand = self._sample_object() 

        self.goal1 = self._sample_goal() + rand2                
        self.goal2 = self.goal1 + self.ee_z_position
        self.goal = np.concatenate([self.goal1, self.goal2])

        self.sim.set_base_pose("vial_target", self.goal1, np.array(self.q)) 
        self.sim.set_base_pose("rack_initial",  self.rack_initial + rand1, np.array(self.q))
        self.sim.set_base_pose("rack_holder_initial",  self.rack_holder1 + rand1, np.array(self.q))

        self.sim.set_base_pose("rack_target",  self.rack_goal + rand2, np.array(self.q))
        self.sim.set_base_pose("rack_holder_target",  self.rack_holder2 + rand2, np.array(self.q))

        Initial_positions = [self.vial_initial_positions[self.objrand - 1] + rand1] + \
        [self.vial_initial_positions[i] + rand1 for i in range(6) if i != self.objrand - 1]
        
        Initial_names = ["vial_to_pick"] + [f"vial{i + 1}" for i in range(5)]
        for target_name, target_position in zip(Initial_names, Initial_positions):
            self.sim.set_base_pose(target_name, target_position, np.array(self.q))
        

    # ------ Select a random well between six wells of the rack for the target position of the vial ---------
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

        return goal
    
    # ----- Select a random well between six wells of the rack for the initial position of the Vial --------
    def _sample_object(self) -> np.ndarray:

        rand = random.randint(1,6)
        if rand == 1:
            initial_object = self.vial_initial_positions[0]
        elif rand == 2:
            initial_object = self.vial_initial_positions[1]
        elif rand == 3:
            initial_object = self.vial_initial_positions[2]
        elif rand == 4:
            initial_object = self.vial_initial_positions[3] 
        elif rand == 5:
            initial_object = self.vial_initial_positions[4]
        else:
            initial_object = self.vial_initial_positions[5]
        return initial_object , rand

     # ---------------------- Success Flag ------------------------------------------
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:

        vial_bodies = [
            'vial1', 'vial2', 'vial3', 'vial4', 'vial5'
        ]
        
        vial_contact_counts = [len(p.getContactPoints(
            bodyA=self.sim._bodies_idx[vial],
            bodyB=self.sim._bodies_idx['vial_to_pick'],
        )) for vial in vial_bodies]

        if any(vial_contact_count != 0 for vial_contact_count in vial_contact_counts):
             penalty = 0.1
        else:
             penalty = 0.0

        d = distance(achieved_goal, desired_goal)
        self.Insert_Flag = np.array(d < self.distance_threshold, dtype=bool)
        return np.array((d + penalty) < self.distance_threshold, dtype=bool)


    # -------------------- Compute the Reward --------------------------------------------------
    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:


        contact_point1 = p.getContactPoints(
            bodyA=self.sim._bodies_idx['vial_to_pick'],
            bodyB=self.sim._bodies_idx['rack_initial'],
            )

        vial_bodies = [
            'vial1', 'vial2', 'vial3', 'vial4', 'vial5'
        ]
        
        vial_contact_counts = [len(p.getContactPoints(
            bodyA=self.sim._bodies_idx[vial],
            bodyB=self.sim._bodies_idx['vial_to_pick'],
        )) for vial in vial_bodies]
        
        d = distance(achieved_goal, desired_goal)

        if len(contact_point1) != 0:
            reward = -1.2
        else:
            #reward = -np.array(d > self.distance_threshold , dtype=np.float64) 
            reward = -d
    
            if any(vial_contact_count != 0 for vial_contact_count in vial_contact_counts):
                reward = reward - 0.5

            if self.Insert_Flag == True:
                if all(vial_contact_count == 0 for vial_contact_count in vial_contact_counts):
                    reward = 5.0
                else:
                    reward = -5.0

        return reward