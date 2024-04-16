import gym
import panda_chemistry_gym
import custom_envs
import numpy as np
from panda_chemistry_gym.pybullet import PyBullet

# choose from the environments below
'''
        'Capping', 
        'Spoon', 

        'EmptyRackInsertion',
        'LoadedRackInsertion',

        'VialInsertion_From_SingleHolder_to_SingleHolder',
        'VialInsertion_From_SingleHolder_to_Rack',
        'VialInsertion_From_SingleHolder_to_LoadedRack',

        'VialInsertion_From_Rack_to_Rack',
        'VialInsertion_From_LoadedRack_to_Rack',
        'VialInsertion_From_Rack_to_LoadedRack',
'''
env = gym.make('Capping-v1', render=True)

obs = env.reset()

done = False
for i in range(1000):
    #action = env.action_space.sample()
    action = np.zeros(np.shape(env.action_space))
    obs, reward, done, info = env.step(action)
    env.render()

    if done:
        env.reset()
