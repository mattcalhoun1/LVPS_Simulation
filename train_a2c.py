import gymnasium
from gymnasium.envs.registration import register
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.autoreset import AutoResetWrapper
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
     max_episode_steps=1000,
)

# this is not ready to use yet
class TrainA2C:
    def __init__(self, model_dir):
        self.__model_dir = model_dir
        self.__max_episode_steps = 100#1000 # max steps per episode
        self.__max_test_steps = 100#1000 # max steps per episode
        self.__max_total_steps = 100#1_000_000
        self.__test_episodes = 5

        # create new instances of the environment
        self.__create_environments('lvps/Search-v0')

        self.__eval_callback = None

    def __create_environments (self, env_id):
        self.__base_env = make_vec_env('lvps/Search-v0', n_envs=4, wrapper_class=self.__wrap_env)
        self.__eval_env = make_vec_env('lvps/Search-v0', n_envs=4, wrapper_class=self.__wrap_env)
        self.__test_env = self.__wrap_env(gymnasium.make(env_id))

    def __wrap_env (self, plain_env):
        return AutoResetWrapper(TimeLimit(plain_env, self.__max_episode_steps))

    def __create_empty_model (self, base_env):
        return A2C(
            policy = "CnnPolicy",#"MlpPolicy",
            env = base_env,
            gamma = 0.98, # original 0.98
            verbose=1,
            seed=1,
            tensorboard_log=f'{self.__model_dir}/tensorboard_log/'
        )


    def train (self):
        logging.getLogger(__name__).info ("Training a new A2C agent...")

        # clean up previous training
        if os.path.exists(f'{self.__model_dir}/evaluation/'):
            shutil.rmtree(f'{self.__model_dir}/evaluation/')
        if os.path.exists(f'{self.__model_dir}/final/'):
            shutil.rmtree(f'{self.__model_dir}/final/')

        # create a new empty model
        model = self.__create_empty_model(self.__base_env)
        self.__recreate_eval_callback(self.__eval_env)

        model = model.learn(total_timesteps=self.__max_total_steps, callback=self.__eval_callback, log_interval=1, progress_bar=True)
        model.save(f'{self.__model_dir}/final/model')

    def continue_training(self):
        model = A2C.load(f'{self.__model_dir}/evaluation/best_model.zip')
        self.__recreate_eval_callback(self.__eval_env)

        model.learn(
            total_timesteps=self.__max_total_steps,
            callback=self.__eval_callback,
            log_interval=50,
            progress_bar=True,
            reset_num_timesteps=True
        )
        model.save(f'{self.__model_dir}/final/model')


    def test_best (self):
        logging.getLogger(__name__).info ("Testing agent...")
        best_model = A2C.load(f'{self.__model_dir}/evaluation/best_model.zip', env=self.__test_env)
        sb3_agent = SB3Agent(best_model)

        _ = evaluate(self.__test_env, sb3_agent, gamma=1.0, episodes=self.__test_episodes, max_steps=self.__max_test_steps, seed=1, show_report=True)

    def test_final (self):
        logging.getLogger(__name__).info ("Testing final agent...")
        best_model = A2C.load(f'{self.__model_dir}/final/model', env=self.__test_env)
        sb3_agent = SB3Agent(best_model)

        _ = evaluate(self.__test_env, sb3_agent, gamma=1.0, episodes=self.__test_episodes, max_steps=self.__max_test_steps, seed=1, show_report=True)

    def __recreate_eval_callback(self, environment):
        self.__eval_callback = EvalCallback(
            environment, eval_freq=1000,
            best_model_save_path=f'{self.__model_dir}/evaluation/',
            log_path=f'{self.__model_dir}/evaluation/',
            warn=False,
            n_eval_episodes=3
        )        

    #def check_env (self):
    #    check_env(self.__lvps_gym_env)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(module)s:%(message)s', level=logging.INFO)
    train = TrainA2C('/home/matt/projects/LVPS_Simulation/models')

    train.train()
    train.test_final()




