import gymnasium as gym
import numpy as np
from gymnasium import spaces
from lvps.strategies.agent_actions import AgentActions
from lvps.strategies.reasonable_search_strategy import ReasonableSearchStrategy
from lvps.simulation.lvps_sim_environment import LvpsSimEnvironment
from lvps.simulation.simulated_agent import SimulatedAgent
from lvps.simulation.agent_types import AgentTypes
from lvps.simulation.sim_events import SimEventType
from field.field_renderer import FieldRenderer
from .lvps_gym_rewards import LvpsGymRewards
import random
import numpy as np
import logging

class LvpsGymEnv(gym.Env):

    metadata = {"render_modes": ["console"]}

    def __init__(self):
        super().__init__()

        self.__scaled_map_height = 400
        self.__scaled_map_width = 400

        self.__observation_image_height_inches = 4
        self.__observation_image_width_inches = 4
        self.__observation_image_dpi = 100

        # Define action and observation space
        self.action_space = spaces.Discrete(AgentActions.NumTrainableActions)

        # observation space is field rendered images as np arrays
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(3, self.__scaled_map_height, self.__scaled_map_width), dtype=np.uint8)

        self.__num_drone_agents = 0
        self.__drone_agents = []
        self.__drone_strategies = {}
        self.__training_agent = None
        self.__lvps_env = None
        self.__found_targets = []
        self.__next_agent_id = 0

        self.__num_targets = 2

    def step(self, action):
        beginning_targets_found = len(self.__found_targets)

        # the 
        #This agent goes
        action_method = self.__get_action_method(self.__training_agent, action)
        action_result = action_method()
        agent_step_targets_found = len(self.__found_targets)


        # each other agent performs a step, according to their strategy
        for d in self.__drone_agents:
            # get a strategy for the action

            pass
        complete_step_targets_found = len(self.__found_targets)

        reward = LvpsGymRewards(self.__training_agent).calculate_reward(
            action_performed=action,
            action_result=action_result,
            target_found=len(complete_step_targets_found) > len(beginning_targets_found),
            target_found_by_this_agent=len(agent_step_targets_found) > len(beginning_targets_found),
            all_targets_found=len(self.__found_targets) == self.__num_targets
        )

        terminated = len(self.__found_targets) == self.__num_targets
        truncated = False
        info = {}

        return self.__get_agent_observation(self.__training_agent), reward, terminated, truncated, info


    def __get_action_method (self, agent, action_num):
        action_map = {
            AgentActions.EstimatePosition : agent.estimate_position,
            AgentActions.Look : agent.look,
            AgentActions.Photograph : agent.photograph,
            AgentActions.Nothing : agent.do_nothing,
            AgentActions.ReportFound : agent.report_found,
            AgentActions.GoForwardShort : agent.go_forward_short,
            AgentActions.GoForwardMedium : agent.go_forward_medium,
            AgentActions.GoForwardFar : agent.go_forward_far,
            AgentActions.GoReverseShort : agent.go_reverse_short,
            AgentActions.GoReverseMedium : agent.go_reverse_medium,
            AgentActions.GoReverseFar : agent.go_reverse_far,
            AgentActions.RotateLeftSmall : agent.rotate_left_small,
            AgentActions.RotateLeftMedium : agent.rotate_left_medium,
            AgentActions.RotateLeftBig : agent.rotate_left_big,
            AgentActions.RotateRightSmall : agent.rotate_right_small,
            AgentActions.RotateRightMedium : agent.rotate_right_medium,
            AgentActions.RotateRightBig : agent.rotate_right_big
        }

        return action_map[action_num]


    def reset(self, seed=None, options=None):
        # create a new LVPS simulation
        self.__lvps_env = None
        self.__found_targets = []
        self.__drone_agents = []
        self.__drone_strategies = {}
        self.__training_agent = None
        self.__next_agent_id = 0

        # add all agents
        self.__add_agents()

        # add targets
        self.__add_targets()

        # subscribe to important events
        self.get_lvps_environment().add_event_subscription (event_type = SimEventType.AgentMoved, listener = self)
        self.get_lvps_environment().add_event_subscription (event_type = SimEventType.TargetFound, listener = self)

        # any auxilary/debugging/etc info to be carried forward
        info = {}

        return self.__get_agent_observation(self.__training_agent), info

    def render(self):
        pass

    def close(self):
        pass

    def __get_agent_observation (self, agent):
        return agent.get_field_renderer().render_field_image_to_array(
            add_game_state=True,
            agent_id=agent.get_id(),
            other_agents_visible=True,
            width_inches=self.__observation_image_width_inches,
            height_inches=self.__observation_image_height_inches,
            dpi=self.__observation_image_dpi)

    def __create_and_add_single_agent (self, field_renderer):
        lvps_x,lvps_y = self.get_lvps_environment().get_field_image_scaler().get_random_traversable_coords()
        lvps_heading = random.randrange(-1800,1800)/10 # pick a random starting heading
        new_agent_id = self.__get_unique_id()

        # this agent receives a paired agent simulation
        lvps_agent = SimulatedAgent(
            agent_id=new_agent_id, 
            agent_type=np.random.choice([AgentTypes.MecCar, AgentTypes.Tank]),
            field_renderer=field_renderer,
            lvps_env=self.get_lvps_environment())

        # add the agent to the lvps simulation
        self.get_lvps_environment().add_agent (lvps_agent, lvps_x, lvps_y, lvps_heading)
        return lvps_agent

    def __add_agents (self):
        field_renderer = FieldRenderer(field_map = self.get_lvps_environment().get_map(), map_scaler=self.get_lvps_environment().get_field_image_scaler())

        # add the agent in training
        self.__training_agent = self.__create_and_add_single_agent(field_renderer=field_renderer)

        # add the drone agents and their strategies
        for i in range(self.__num_drone_agents):
            drone = self.__create_and_add_single_agent(field_renderer=field_renderer)
            self.__drone_agents.append(drone)
            self.__drone_strategies[drone.get_id()] = ReasonableSearchStrategy(render_field=False)

    def __add_targets (self):
        # create targets
        for i in range(self.__num_targets):
            lvps_x,lvps_y = self.get_lvps_environment().get_field_image_scaler().get_random_traversable_coords()
            target_id = self.__get_unique_id()
            self.get_lvps_environment().add_target(
                target_id=target_id,
                target_name=f'coin_{target_id}',
                target_type='coin',
                target_x=lvps_x,
                target_y=lvps_y)

    def handle_event (self, event_type, event_details):
        if event_type == SimEventType.AgentMoved:
            # update the view
            agent_id = event_details['agent_id']
            lvps_x = event_details['x']
            lvps_y = event_details['y']
            heading = event_details['heading']
            logging.getLogger(__name__).info(f"Gym Env notified agent {agent_id} moved to position: {round(lvps_x)},{round(lvps_y)}, heading: {heading}")
            
        elif event_type == SimEventType.TargetFound:
            target_id = event_details['target_id']
            if target_id not in self.__found_targets:
                self.__found_targets.append(target_id)

            if len(self.__found_targets) >= self.__num_targets:
                logging.getLogger(__name__).info("All targets found. search is complete")

    def get_lvps_environment (self):
        if self.__lvps_env is None:
            self.__lvps_env = LvpsSimEnvironment()
        return self.__lvps_env

    def __get_total_distance_traveled (self):
        total = 0
        for a in self.__drone_agents:
            agent = self.__drone_agents[a]
            total += agent.get_total_distance_traveled()

        total += self.__training_agent.get_total_distance_traveled()
        return total
    
    def __get_unique_id (self):
        self.__next_agent_id += 1
        return self.__next_agent_id
