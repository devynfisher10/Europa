
import numpy as np
from rlgym.envs import Match
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy


from europa_obs import EuropaObs
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogRewardCallback


from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward, VelocityReward
from rewards import VelocityBallToGoalReward, KickoffReward, TouchVelChange,JumpTouchReward,DoubleTapReward 
from rewards import AirDribbleReward, AerialReward, GoalSpeedAndPlacementReward, BoostInAir, ExploreAir



# using Leaky Relu activation function
from torch.nn import LeakyReLU

from state import EuropaStateSetter
from terminal import EuropaTerminalCondition
from action_parser import LookupAction

if __name__ == '__main__':  # Required for multiprocessing
    testing = False # if testing new changes, set instances to 1 and gamespeed to 1
    model_name = 'EuropaRun2'
    loading_model=True #check that loading model instead of starting from scratch

    if testing:
        num_instances=1
        game_speed=1
    else:
        num_instances=8
        game_speed=100
    frame_skip = 8          # Number of ticks to repeat an action
    half_life_seconds = 6   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    # initial parameters, will update over the course of learning to increase batch size and comment iterations
    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  
    target_steps = 500_000 # from 100,000 at 1B
    agents_per_match = 2
    steps = target_steps // (num_instances * agents_per_match) #making sure the experience counts line up properly
    batch_size = 100_000 # from 20,000 at 1B
    model_to_load = "exit_save.zip" #exit_save.zip rl_model_281069544_steps
    print(f"fps={fps}, gamma={gamma})")



    def get_match():  
        goal_weight = 10 
        demo_weight = 3 
        boost_weight = .25 

        # defining initial custom reward weights, will update over time for curriculum learning and comment iterations
        event_weight = 1 
        touch_vel_weight = 11 # from 14 at 1B
        vel_ball_weight = 3 # 
        vel_weight = .002 
        jump_touch_weight = 25 
        double_tap_weight = 1  
        air_dribble_weight = 2.5 
        aerial_weight = .00075 # from .0005 at 1B  
        goal_speed_weight = 10 
        kickoff_weight = .3 
        boost_in_air_weight = .3
        explore_air_weight = .3

        return Match(
            team_size=1,  # 1v1 bot only
            tick_skip=frame_skip,
            ##### set to use EuropaRewards
            reward_function=SB3CombinedLogReward(
                (
                 EventReward(goal=goal_weight, concede=-goal_weight, demo=demo_weight, boost_pickup=boost_weight),  
                 TouchVelChange(),
                 VelocityBallToGoalReward(),
                 VelocityReward(),
                 JumpTouchReward(min_height=92.75*2), # from 92.75 at 2B from 0 at 1.25B
                 DoubleTapReward(),
                 AirDribbleReward(),
                 AerialReward(),
                 GoalSpeedAndPlacementReward(),
                 KickoffReward(),
                 BoostInAir(),
                 ExploreAir(),
                 ),
                (event_weight, touch_vel_weight, vel_ball_weight, vel_weight, jump_touch_weight, 
                    double_tap_weight, air_dribble_weight, aerial_weight, goal_speed_weight, kickoff_weight,
                    boost_in_air_weight, explore_air_weight),
                "logs"
            ),
            spawn_opponents=True,
            terminal_conditions=[EuropaTerminalCondition()],
            obs_builder=EuropaObs(),  
            state_setter=EuropaStateSetter(),  
            action_parser=LookupAction(),
            game_speed=game_speed 
        )

    # creating env and force paging 
    env = SB3MultipleInstanceEnv(get_match, num_instances, wait_time=50, force_paging=True)            
    env = VecCheckNan(env)                                # Optional
    env = VecMonitor(env)                                 # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

    # we try loading existing model and if does not exist then we create new model
    try:
        assert loading_model
        model=PPO.load(
            f"models/{model_to_load}",
            env,
            device="auto",
            custom_objects={"n_envs": env.num_envs, "n_steps":steps, "batch_size":batch_size}, #automatically adjusts to users changing instance count, may encounter shaping error otherwise. need to modify hyperparams here for loading model
            )
        print("loaded model")
        print(f"Current timesteps: {model.num_timesteps}")
    except:
        print("creating model")
        
        policy_kwargs = dict(
            activation_fn=LeakyReLU,
            net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 400, 256])
            
         )
        model = PPO(
            MlpPolicy,
            env,
            n_epochs=30,                 
            policy_kwargs=policy_kwargs,
            learning_rate=1e-4,          
            ent_coef=0.01,               # From PPO Atari
            vf_coef=1.,                  # From PPO Atari
            gamma=gamma,                 # Gamma as calculated using half-life
            clip_range=0.2,
            verbose=3,                   # Print out all the info as we're going
            batch_size=batch_size,             
            n_steps=steps,                # Number of steps to perform before optimizing network
            tensorboard_log="logs",  
            device="auto",           # Uses GPU if available
            pivotal_state_replay=True # uses pivotal state replay alteration to PPO
        )
    print("running")

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name

    reward_list=['event', 'touch_vel','vel_ball','vel','jump_touch','double_tap', 'air_dribble', 'aerial', 'goal_speed_and_placement', 'kickoff', 'boost_in_air', 'explore_air']

    save_callback = CheckpointCallback(round(5_000_000 / env.num_envs), save_path="models", name_prefix=model_name)
    reward_callback = SB3CombinedLogRewardCallback(reward_list, 'logs')
    callbacks = CallbackList([save_callback, reward_callback])
    try:
        while True:
            model.learn(100_000_000, callback=callbacks, reset_num_timesteps=False, tb_log_name=model_name)

            model.save("models/exit_save")
            model.save("mmr_models/" + str(model.num_timesteps))
    
    except KeyboardInterrupt:
        model.save("models/exit_save")
        model.save("mmr_models/" + str(model.num_timesteps))
        print("Exiting training")


