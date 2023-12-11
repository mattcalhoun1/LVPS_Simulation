from gymnasium.envs.registration import register
import logging
import torch
from .agent_actions import AgentActions
from lvps.simulation.lvps_sim_environment import LvpsSimEnvironment
from lvps.simulation.simulated_agent import SimulatedAgent
from lvps.gym.lvps_gym_env import LvpsGymEnv
import gymnasium
import numpy as np
from .agent_strategy import AgentStrategy
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.policies import obs_as_tensor
import torch

register(
     id="lvps/Search-v0",
     entry_point="lvps.gym.lvps_gym_env:LvpsGymEnv",
     max_episode_steps=1000,
)

class RLSearchStrategy(AgentStrategy):
    def __init__(self, environment : LvpsSimEnvironment, model_file : str):
        self.__environment = environment
        self.__model_file = model_file
        #self.__rl_model = DQN.load(model_file)#, env=self.__environment)

        #self.__create_environments()
        #env = gymnasium.make('lvps/Search-v0')

        self.__device = 'cuda' # cpu or cuda
        self.__gym_env = LvpsGymEnv()
        self.__gym_env_initialized = False

        self.__rl_model = DQN.load(model_file)#, device=device, print_system_info=True, env=LvpsGymEnv(), force_reset=True)
        self.__rl_model.set_logger(logging.getLogger(__name__))
        self.__gym_env.reset()
        #self.__rl_model.batch_size = 1
        self.__observation_image_height_inches = 4
        self.__observation_image_width_inches = 4
        self.__observation_image_dpi = 100       

    def __predict_proba(self, model, obs):
        #obs_tensor = obs_as_tensor(obs, model.policy.device)
        #dis = model.policy.get_distribution(obs)

        reordered_obs = np.transpose(obs, (2,0,1))

        logging.getLogger(__name__).info(f"Reordered shape: {reordered_obs.shape}")
        logging.getLogger(__name__).info(f"cnn model obs space: {model.policy.observation_space}")

        probs, _ = model.policy.predict(reordered_obs, deterministic = False)
        return probs

    def get_next_action (self, lvps_agent : SimulatedAgent, last_action, last_action_result, step_count):
        logging.getLogger(__name__).info(f"Determining next action for agent {lvps_agent.get_id()}")

        # estimate position
        lvps_agent.estimate_position()
        logging.getLogger(__name__).info(f"Agent at: {lvps_agent.get_last_coords_and_heading()}")

        lvps_agent.get_field_renderer().save_field_image(
            image_file='/tmp/lvpssim/rl.png',
            add_game_state=True,
            agent_id=lvps_agent.get_id(),
            other_agents_visible=True,
            width_inches=self.__observation_image_width_inches,
            height_inches=self.__observation_image_height_inches,
            dpi=self.__observation_image_dpi
        )

        # get the agent's perspecive rendering as the observation
        obs = lvps_agent.get_field_renderer().render_field_image_to_array(
            add_game_state=True,
            agent_id=lvps_agent.get_id(),
            other_agents_visible=True,
            width_inches=self.__observation_image_width_inches,
            height_inches=self.__observation_image_height_inches,
            dpi=self.__observation_image_dpi
        ).copy()#.reshape(3, 400, 400)
        #logging.getLogger(__name__).info(f"Observation shape: {obs.shape}, {obs.dtype}")

        #for r in range(400):
        #    for c in range(400):
        #        if obs[r][c] != 0 and obs[r][c] != 255:
        #            logging.getLogger(__name__).info(f"{obs[r][c]}")

        #obs_tensor = torch.from_numpy(obs)

        #logging.getLogger(__name__).info(f"action space: {self.__rl_model.get_env().action_space}")
        #logging.getLogger(__name__).info(f"observation space: {self.__rl_model.get_env().observation_space}")
        #logging.getLogger(__name__).info(f"observation: {obs[250][250]}")

        probs = self.__predict_proba(self.__rl_model, obs)
        logging.getLogger(__name__).info(f"RL Probs: {AgentActions.Names[probs.argmax()]}")

        action, _states = self.__rl_model.predict(obs, {}, deterministic = True)
        #logging.getLogger(__name__).info(f"RL Model returned type: {type(action)}, shape: {action.shape} -  {action}, {_states}")        
        selected_action = action
        if type(action) is np.array or type(action) is np.ndarray:
            selected_action = np.argmax(action)
        lvps_agent.estimate_position()
        logging.getLogger(__name__).info(f"Agent selected action {AgentActions.Names[selected_action]}")

        return selected_action, {}

        #if len(action.shape) > 0:
        #    selected_action = action[action.argmax]

        #    logging.getLogger(__name__).info(f"RL Model chooses action {AgentActions.Names[selected_action]}")

        #    return selected_action, {}
        #else:
        #    logging.getLogger(__name__).info("Model gave us nothing.")
        #    return AgentActions.Nothing, {}
