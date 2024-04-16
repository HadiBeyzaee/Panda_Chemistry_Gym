import os
import sys

sys.path.append("..")

from gym.envs.registration import register

for reward_type in ["sparse", "dense"]:
    for control_type in ["ee", "joints"]:
        reward_suffix = "Dense" if reward_type == "dense" else ""
        control_suffix = "Joints" if control_type == "joints" else ""
        kwargs = {"reward_type": reward_type, "control_type": control_type}
#--------------------------------------------------------------------------------
        register(
            id="Capping{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:Capping_Env",
            kwargs=kwargs,
            max_episode_steps=100,
        )

#---------------------------------------------------------------------------------
        register(
            id="EmptyRackInsertion{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:EmptyRackInsertion_Env",
            kwargs=kwargs,
            max_episode_steps=150,
        )

        register(
            id="LoadedRackInsertion{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:LoadedRackInsertion_Env",
            kwargs=kwargs,
            max_episode_steps=100,
        )

#-------------------------------------------------------------------------------------
        register(
            id="VialInsertion_From_LoadedRack_to_Rack{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:VialInsertion_From_LoadedRack_to_Rack_Env",
            kwargs=kwargs,
            max_episode_steps=100,
        )
        register(
            id="VialInsertion_From_Rack_to_Rack{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:VialInsertion_From_Rack_to_Rack_Env",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id="VialInsertion_From_Rack_to_LoadedRack{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:VialInsertion_From_Rack_to_LoadedRack_Env",
            kwargs=kwargs,
            max_episode_steps=100,
        )
        register(
            id="VialInsertion_From_SingleHolder_to_Rack{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:VialInsertion_SingleHolder_Rack_Env",
            kwargs=kwargs,
            max_episode_steps=100,
        )
        register(
            id="VialInsertion_From_SingleHolder_to_LoadedRack{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:VialInsertion_SingleHolder_LoadedRack_Env",
            kwargs=kwargs,
            max_episode_steps=100,
        )
        register(
            id="VialInsertion_From_SingleHolder_to_SingleHolder{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:VialInsertion_SingleHolder_SingleHolder_Env",
            kwargs=kwargs,
            max_episode_steps=100,
        )
        register(
            id="Spoon{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:Spoon_Env",
            kwargs=kwargs,
            max_episode_steps=100,
        )
        register(
            id="Spoon_Vial{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:Spoon_Vial_Env",
            kwargs=kwargs,
            max_episode_steps=100,
        )
        register(
            id="Spoon_Horizental{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:Spoon_Horizental_Env",
            kwargs=kwargs,
            max_episode_steps=100,
        )
        register(
            id="Spoon_Real{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:Spoon_Real_Env",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id="Spoon_Real2{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:Spoon_Real_Env2",
            kwargs=kwargs,
            max_episode_steps=100,
        )
        
        register(
            id="Reach{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:Reach_Env",
            kwargs=kwargs,
            max_episode_steps=100,
        )
        