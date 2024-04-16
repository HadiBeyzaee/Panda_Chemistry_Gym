import argparse
import os
import gym
import custom_envs
from stable_baselines3 import SAC, DDPG, TD3
from sb3_contrib import TQC


def main():
    parser = argparse.ArgumentParser(description="Load and visualize a trained RL model")
    parser.add_argument("--env_id", type=str, required=True, help="Environment ID")
    parser.add_argument("--algorithm_name", type=str, required=True, help="Algorithm name")
    args = parser.parse_args()

    # Create the environment
    env = gym.make(args.env_id + '-v1', render=True)

    # Define the model path based on convention (without algorithm name as a suffix)
    model_path = f"./trained_model/{args.env_id}/{args.env_id}_{args.algorithm_name}"

    # Load the trained model
    if args.algorithm_name == 'DDPG':
        model = DDPG.load(model_path, env=env)
    elif args.algorithm_name == 'SAC':
        model = SAC.load(model_path, env=env)
    elif args.algorithm_name == 'TQC':
        model = TQC.load(model_path, env=env)
    elif args.algorithm_name == 'TD3':
        model = TQC.load(model_path, env=env)
    else:
        raise ValueError("Unsupported algorithm name")

    obs = env.reset()

    for i in range(2000):
       # print('First obs: ', obs)
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


if __name__ == "__main__":
    main()

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

# Example
# python show_main.py --env_id 'VialInsertion_From_SingleHolder_to_SingleHolder' --algorithm_name 'TD3'