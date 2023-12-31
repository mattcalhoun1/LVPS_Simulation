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

    def __init__(self, render_mode=None):
        super().__init__()

        self.__scaled_map_height = 400
        self.__scaled_map_width = 400
        self.__grayscale = True

        self.__observation_image_height_inches = 4
        self.__observation_image_width_inches = 4
        self.__observation_image_dpi = 100

        # Define action and observation space
        self.action_space = spaces.Discrete(AgentActions.NumTrainableActions)

        # observation space is field rendered images as np arrays
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.__scaled_map_height, self.__scaled_map_width, 1 if self.__grayscale else 3), dtype=np.uint8)

        self.__training_agent = None
        self.__reward_calculator = None
        self.__lvps_env = None
        self.__found_targets = []
        self.__next_agent_id = 0
        self.__truncate_on_out_bounds = True

        # drone agent state
        self.__num_drone_agents = 0
        self.__drone_agents = []
        self.__drone_strategies = {}

        # These are required for the hardcoded strategies
        self.__drone_last_action = {}
        self.__drone_last_result = {}

        self.__num_targets = 2
        self.__lvps_sim_step = 0
        self.__reset_count = 0

    def step(self, action):
        self.__lvps_sim_step += 1
        beginning_targets_found = len(self.__found_targets)
        beg_nearest_unfound_target_id, beg_nearest_unfound_target_dist, beg_nearest_unfound_heading = self.__training_agent.get_nearest_unfound_target_distance()

        # This agent goes
        self.__update_agent_coords(agent=self.__training_agent, force_refresh=False)
        action_method = self.__get_action_method(self.__training_agent, action)
        action_result = action_method()
        self.__execute_auto_follow_ups(
            agent = self.__training_agent,
            latest_action=action,
            latest_result=action_result
        )
        
        self.__update_agent_coords(agent=self.__training_agent, force_refresh=True)
        agent_step_targets_found = len(self.__found_targets)
        end_nearest_unfound_target_id, end_nearest_unfound_target_dist, end_nearest_unfound_heading = self.__training_agent.get_nearest_unfound_target_distance()


        # each other agent performs a step, according to their strategy
        for d in self.__drone_agents:
            # get a strategy for the action
            agent_strategy = self.__drone_strategies[d.get_id()]
            drone_last_action = None if d.get_id() not in self.__drone_last_action else self.__drone_last_action[d.get_id()]
            drone_last_result = None if d.get_id() not in self.__drone_last_result else self.__drone_last_result[d.get_id()]
            self.__update_agent_coords(agent=d, force_refresh=False)

            next_drone_action, next_drone_params = agent_strategy.get_next_action (
                lvps_agent = d,
                last_action = drone_last_action,
                last_action_result = drone_last_result,
                step_count=self.__lvps_sim_step)
            
            drone_method, next_drone_params = self.__get_drone_action_method_and_params (
                drone=d,
                action_num = next_drone_action,
                action_params=next_drone_params
            )
            drone_result = None
            if next_drone_params is not None:
                drone_result = drone_method(next_drone_params)
            else:
                drone_result = drone_method()
            self.__execute_auto_follow_ups(
                agent = d,
                latest_action=next_drone_action,
                latest_result=drone_result
            )
            self.__drone_last_action[d.get_id()] = next_drone_action
            self.__drone_last_result[d.get_id()] = drone_result
            self.__update_agent_coords(agent=d, force_refresh=True)

        complete_step_targets_found = len(self.__found_targets)

        truncated = False
        if self.__truncate_on_out_bounds:
            # if the agent is out of bounds or in an obstacle, end the episode
            if self.__training_agent.is_out_of_bounds() or self.__training_agent.is_in_obstacle():
                truncated = True


        reward = self.__reward_calculator.calculate_reward(
            action_performed=action,
            action_result=action_result,
            target_found=complete_step_targets_found > beginning_targets_found,
            target_found_by_this_agent=agent_step_targets_found > beginning_targets_found,
            all_targets_found=len(self.__found_targets) == self.__num_targets,
            beg_nearest_unfound_target_id=beg_nearest_unfound_target_id,
            beg_nearest_unfound_target_dist=beg_nearest_unfound_target_dist,
            end_nearest_unfound_target_id=end_nearest_unfound_target_id,
            end_nearest_unfound_target_dist=end_nearest_unfound_target_dist,
            is_within_photo_distance=end_nearest_unfound_target_dist is not None and end_nearest_unfound_target_dist <= self.__training_agent.get_photo_distance()
        )

        self.__execute_auto_follow_ups(
            agent = self.__training_agent,
            latest_action=action,
            latest_result=action_result
        )

        terminated = len(self.__found_targets) == self.__num_targets
        info = {}

        if reward > 0:
            logging.getLogger(__name__).info(f"Action: {self.__get_action_name(action)}, Result: {action_result}, Reward: {reward}")

        return self.__get_agent_observation(self.__training_agent), reward, terminated, truncated, info
    
    def __get_action_name(self, action):
        if type(action) is np.array or type(action) is np.ndarray:
            return AgentActions.Names[action.max()]
        return AgentActions.Names[action]        

    def __update_agent_coords (self, agent : SimulatedAgent, force_refresh : bool):
        last_x, last_y, last_heading, last_conf = agent.get_last_coords_and_heading()
        if force_refresh or last_x is None:
            agent.estimate_position()

    def __get_action_method (self, agent : SimulatedAgent, action_num):
        action_map = {
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

        if type(action_num) is np.array or type(action_num) is np.ndarray:
            return action_map[action_num.max()]
        
        return action_map[action_num]

    # drone has an extended set of actions
    def __get_drone_action_method_and_params (self, drone : SimulatedAgent, action_num, action_params):
        action_map = {
            AgentActions.EstimatePosition : drone.estimate_position,
            AgentActions.Look : drone.look,
            AgentActions.Photograph : drone.photograph,
            AgentActions.Nothing : drone.do_nothing,
            AgentActions.ReportFound : drone.report_found,
            AgentActions.GoForwardShort : drone.go_forward_short,
            AgentActions.GoForwardMedium : drone.go_forward_medium,
            AgentActions.GoForwardFar : drone.go_forward_far,
            AgentActions.GoReverseShort : drone.go_reverse_short,
            AgentActions.GoReverseMedium : drone.go_reverse_medium,
            AgentActions.GoReverseFar : drone.go_reverse_far,
            AgentActions.RotateLeftSmall : drone.rotate_left_small,
            AgentActions.RotateLeftMedium : drone.rotate_left_medium,
            AgentActions.RotateLeftBig : drone.rotate_left_big,
            AgentActions.RotateRightSmall : drone.rotate_right_small,
            AgentActions.RotateRightMedium : drone.rotate_right_medium,
            AgentActions.RotateRightBig : drone.rotate_right_big,
            AgentActions.Go : drone.go,
            AgentActions.Rotate : drone.rotate,
            AgentActions.GoForward : drone.go_forward,
            AgentActions.GoReverse : drone.go_reverse,
            AgentActions.AdjustRandomly : drone.adjust_randomly

        }

        actions_with_params = [
            AgentActions.Go,
            AgentActions.Rotate,
            AgentActions.GoForward,
            AgentActions.GoReverse,
            AgentActions.Strafe,
        ]
        filtered_action_params = None
        safe_action_num = action_num

        if type(action_num) is np.array or type(action_num) is np.ndarray:
            safe_action_num = action_num.max()

        if safe_action_num in actions_with_params:
            filtered_action_params = action_params


        return action_map[safe_action_num], filtered_action_params

    def __execute_auto_follow_ups (self, agent : SimulatedAgent, latest_action, latest_result):
        if type(latest_action) is np.array or type(latest_action) is np.ndarray:
            latest_action = latest_action.max()
                
        if latest_action == AgentActions.Photograph and latest_result == True:
            logging.getLogger(__name__).info(f"Auto-executing target found report for agent {agent.get_id()}")
            # a successful photograph was taken, report it
            report_success = False
            retries = 0
            max_retries = 5
            while (report_success == False and retries < max_retries):
                report_success = agent.report_found()
                retries += 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # create a new LVPS simulation
        self.__reset_count += 1
        self.__lvps_env = None
        self.__found_targets = []
        self.__drone_agents = []
        self.__drone_strategies = {}
        self.__drone_last_action = {}
        self.__drone_last_result = {}
        self.__reward_calculator = None

        self.__training_agent = None
        self.__next_agent_id = 0
        self.__lvps_sim_step = 0

        # add all agents
        self.__add_agents()

        self.__reward_calculator = LvpsGymRewards(self.__training_agent)

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
        field_renderer = FieldRenderer(
            field_map = self.get_lvps_environment().get_map(),
            map_scaler=self.get_lvps_environment().get_field_image_scaler(),
            grayscale=self.__grayscale)

        # add the agent in training
        self.__training_agent = self.__create_and_add_single_agent(field_renderer=field_renderer)
        logging.getLogger(__name__).info("Added training agent.")

        # add the drone agents and their strategies
        for i in range(self.__num_drone_agents):
            drone = self.__create_and_add_single_agent(field_renderer=field_renderer)
            self.__drone_agents.append(drone)
            self.__drone_strategies[drone.get_id()] = ReasonableSearchStrategy(render_field=False)
            logging.getLogger(__name__).info("Added drone agent.")

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
            logging.getLogger(__name__).debug(f"Gym Env notified agent {agent_id} moved to position: {round(lvps_x)},{round(lvps_y)}, heading: {heading}")
        elif event_type == SimEventType.TargetFound:
            target_id = event_details['target_id']
            if target_id not in self.__found_targets:
                self.__found_targets.append(target_id)

            if len(self.__found_targets) >= self.__num_targets:
                logging.getLogger(__name__).info("All targets found. search is complete")

    def get_lvps_environment (self):
        if self.__lvps_env is None:
            self.__lvps_env = LvpsSimEnvironment(id=self.__reset_count)
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
