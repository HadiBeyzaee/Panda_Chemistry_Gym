import argparse
import gym
import panda_chemistry_gym
from stable_baselines3 import HerReplayBuffer, DDPG, TD3, SAC
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import EvalCallback
import custom_envs  
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

def main(args):
    env_id = args.environment
    log_dir = './tensorboard/' + env_id
    total_timesteps = args.timesteps    

    eval_env = gym.make(env_id + '-v1', render = True)

    eval_callback = EvalCallback(eval_env, best_model_save_path="./best_models_path/" + env_id,
                                  log_path="./log_path/" + env_id, eval_freq=200_000,
                                  deterministic=True, render=False)

    algorithms = {
        'SAC': SAC,
        'TQC': TQC,
        'DDPG': DDPG,
        'TD3': TD3,
    }

    algorithm_cls = algorithms[args.algorithm]

    n_actions = eval_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


    model = algorithm_cls(policy="MultiInputPolicy", 
                      env=eval_env, 
                      action_noise=action_noise,
                      learning_rate=1e-4,  
                      buffer_size=3_000_000, 
                      batch_size=2048, 
                      replay_buffer_class=HerReplayBuffer, 
                      learning_starts = 1000,
                      policy_kwargs=dict(net_arch=[1024, 1024, 1024, 1024], n_critics=3), 
                      gamma=0.95, 
                      tau=0.002, 
                      verbose=2,  
                      tensorboard_log=log_dir)



    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model.save('./trained_model/' + env_id + '/' + env_id + '_' + model.__class__.__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL models on different environments")
    parser.add_argument("--algorithm", type=str, choices=['SAC', 'TQC', 'TD3' ,'DDPG' ],
                        default='SAC', help="RL algorithm to use")
    parser.add_argument("--environment", type=str, choices=[
        'Capping', 
        'Spoon',

        'Reach',

        'EmptyRackInsertion',
        'LoadedRackInsertion',

        'VialInsertion_From_SingleHolder_to_SingleHolder',
        'VialInsertion_From_SingleHolder_to_Rack',
        'VialInsertion_From_SingleHolder_to_LoadedRack',

        'VialInsertion_From_Rack_to_Rack',
        'VialInsertion_From_LoadedRack_to_Rack',
        'VialInsertion_From_Rack_to_LoadedRack',

    ], default='Capping', help="Environment ID")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total timesteps for training")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args, _ = parser.parse_known_args()

    main(args)

    # Example
    # python train_main.py --algorithm 'DDPG' --timesteps 2_000_000 --environment 'EmptyRackInsertion' --seed 454
