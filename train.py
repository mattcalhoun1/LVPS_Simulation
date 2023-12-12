import gymnasium
from gymnasium.envs.registration import register
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.autoreset import AutoResetWrapper
from stable_baselines3.common.env_util import make_vec_env
from lvps.gym.rbean_utils import evaluate, SB3Agent
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
import warnings
warnings.filterwarnings('ignore')
import logging
import os
import shutil

register(
     id="lvps/Search-v0",
     entry_point="lvps.gym.lvps_gym_env:LvpsGymEnv",
     max_episode_steps=1000,
)

class Train:
    def __init__(self, model_dir):
        self.__model_dir = model_dir
        self.__max_episode_steps = 1000 # max steps per episode
        self.__max_test_steps = 1000 # max steps per episode
        self.__max_total_steps = 1_000_000
        self.__test_episodes = 3

        # create new instances of the environment
        self.__create_environments('lvps/Search-v0')

        self.__eval_callback = None

    def __create_environments (self, env_id):
        self.__base_env = AutoResetWrapper(TimeLimit(gymnasium.make(env_id), self.__max_episode_steps))
        self.__eval_env = AutoResetWrapper(TimeLimit(gymnasium.make(env_id), self.__max_episode_steps))
        self.__test_env = AutoResetWrapper(TimeLimit(gymnasium.make(env_id), self.__max_episode_steps))

    def __create_empty_model (self, base_env):

        return DQN(
            policy = "CnnPolicy",
            env = base_env,
            #learning_rate = 4e-3, # original 4e-3

            batch_size = 128, # original 128
            buffer_size = 5000, # original 4k. However, memory is contained in the observation image, so do we need a buffer?
            learning_starts = 0, # original 0

            gamma = 0.98, # original 0.98
            target_update_interval = 600, # original 600
            #train_freq = 16, # original 16
            #gradient_steps = 8, # original 8

            exploration_fraction = 0.2, # original 0.2
            exploration_initial_eps = 1.0, # original 1.0
            exploration_final_eps = 0.05, # original 0.07

            policy_kwargs = dict(net_arch=[128,128,64]),
            verbose=1,
            seed=1,
            tensorboard_log=f'{self.__model_dir}/tensorboard_log/'
        )

    def continue_training(self):
        model = DQN.load(f'{self.__model_dir}/evaluation/best_model.zip')
        self.__recreate_eval_callback(self.__eval_env)

        model.learn(
            total_timesteps=self.__max_total_steps,
            callback=self.__eval_callback,
            log_interval=50,
            progress_bar=True,
            reset_num_timesteps=True
        )
        model.save(f'{self.__model_dir}/final/model')

    def train (self):
        logging.getLogger(__name__).info ("Training a new agent...")

        # clean up previous training
        if os.path.exists(f'{self.__model_dir}/evaluation/'):
            shutil.rmtree(f'{self.__model_dir}/evaluation/')
        if os.path.exists(f'{self.__model_dir}/final/'):
            shutil.rmtree(f'{self.__model_dir}/final/')

        # create a new empty model
        model = self.__create_empty_model(self.__base_env)
        self.__recreate_eval_callback(self.__eval_env)

        model = model.learn(total_timesteps=self.__max_total_steps, callback=self.__eval_callback, log_interval=50, progress_bar=True)
        model.save(f'{self.__model_dir}/final/model')

    def test_best (self):
        logging.getLogger(__name__).info ("Testing agent...")
        best_model = DQN.load(f'{self.__model_dir}/evaluation/best_model.zip', env=self.__test_env)
        sb3_agent = SB3Agent(best_model)

        _ = evaluate(self.__test_env, sb3_agent, gamma=1.0, episodes=self.__test_episodes, max_steps=self.__max_test_steps, seed=1, show_report=True)

    def test_final (self):
        logging.getLogger(__name__).info ("Testing final agent...")
        best_model = DQN.load(f'{self.__model_dir}/final/model', env=self.__test_env)
        sb3_agent = SB3Agent(best_model)

        _ = evaluate(self.__test_env, sb3_agent, gamma=1.0, episodes=self.__test_episodes, max_steps=self.__max_test_steps, seed=1, show_report=True)

    def __recreate_eval_callback(self, environment):
        self.__eval_callback = EvalCallback(
            environment, eval_freq=20000,
            best_model_save_path=f'{self.__model_dir}/evaluation/',
            log_path=f'{self.__model_dir}/evaluation/',
            warn=False,
            n_eval_episodes=2
        )        

    #def check_env (self):
    #    check_env(self.__lvps_gym_env)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(module)s:%(message)s', level=logging.INFO)
    train = Train('/home/matt/projects/LVPS_Simulation/models')

    train.train()
    train.test_final()




