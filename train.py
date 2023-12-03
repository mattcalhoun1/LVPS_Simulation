import gymnasium
from gymnasium.envs.registration import register
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from lvps.gym.rbean_utils import evaluate, SB3Agent
import warnings
warnings.filterwarnings('ignore')
import logging
import os
import shutil

register(
     id="lvps/Search-v0",
     entry_point="lvps.gym.lvps_gym_env:LvpsGymEnv",
     max_episode_steps=10000,
)

class Train:
    def __init__(self, model_dir):
        self.__model_dir = model_dir
        self.__max_episode_steps = 10000 # max steps per episode

        # create new instances of the environment
        self.__create_environments('lvps/Search-v0')

        if os.path.exists(f'{model_dir}/evaluation/'):
            shutil.rmtree(f'{model_dir}/evaluation/')
        self.__eval_callback = None

    def __create_environments (self, env_id):
        self.__base_env = TimeLimit(gymnasium.make(env_id), self.__max_episode_steps)
        #self.__base_envs = make_vec_env(env_id=env_id, n_envs=env_count, seed=0)
        #self.__eval_envs = make_vec_env(env_id=env_id, n_envs=env_count, seed=0)
        self.__test_env = gymnasium.make(env_id, render_mode='console')        

    def __create_empty_model (self, base_env):
        return DQN(
            policy = "CnnPolicy",#"MlpPolicy",
            env = base_env,
            learning_rate = 4e-3, # original 4e-3

            batch_size = 64, # original 128
            buffer_size = 10_000, # original 10k
            learning_starts = 0, # original 0

            gamma = 0.98, # original 0.98
            target_update_interval = 600, # original 600
            train_freq = 16, # original 16
            gradient_steps = 8, # original 8

            exploration_fraction = 0.2, # original 0.2
            exploration_initial_eps = 1.0, # original 1.0
            exploration_final_eps = 0.07, # original 0.07

            #policy_kwargs = dict(net_arch=[256,128,64]),
            verbose=1,
            seed=1
        )


    def train (self):
        logging.getLogger(__name__).info ("Training a new agent...")
        # create a new empty model
        model = self.__create_empty_model(self.__base_env)
        self.__recreate_eval_callback(self.__base_env)

        model = model.learn(total_timesteps=200_000, callback=self.__eval_callback)

    def test (self):
        logging.getLogger(__name__).info ("Testing agent...")
        best_model = DQN.load(f'{self.__model_dir}/evaluation/best_model.zip', env=self.__test_env)
        sb3_agent = SB3Agent(best_model)

        _ = evaluate(self.__test_env, sb3_agent, gamma=1.0, episodes=50, max_steps=200, seed=1)

    def __recreate_eval_callback(self, environment):
        self.__eval_callback = EvalCallback(
            environment, n_eval_episodes=20, eval_freq=1000,
            best_model_save_path=f'{self.__model_dir}/evaluation/', log_path=f'{self.__model_dir}/evaluation/', warn=False
        )        

    #def check_env (self):
    #    check_env(self.__lvps_gym_env)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(module)s:%(message)s', level=logging.INFO)
    train = Train('/home/matt/projects/LVPS_Simulation/models')

    train.train()
    train.test()




