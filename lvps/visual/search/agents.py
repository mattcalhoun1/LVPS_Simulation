import math
import random
import mesa
import logging
#from .field_guide import FieldGuide
import numpy as np
from lvps.simulation.simulated_agent import SimulatedAgent
from lvps.strategies.agent_strategy import AgentStrategy
from lvps.strategies.agent_actions import AgentActions

from field.field_renderer import FieldRenderer
from field.field_scaler import FieldScaler

class SearchAgent(mesa.Agent):
    def __init__(self, unique_id, pos, model, scaler : FieldScaler=None, agent_strategy : AgentStrategy = None, lvps_agent : SimulatedAgent = None):
        super().__init__(unique_id, model)
        self.__scaler = scaler
        self.__model = model
        self.__unique_id = unique_id
        self.__lvps_sim_agent = lvps_agent

        self.pos = pos # position within the VISUAL simulation

        self.__step_count = 0

        self.__curr_action = AgentActions.Nothing
        self.__curr_action_start_time = 0
        self.__curr_action_result = True

        self.__agent_strategy = agent_strategy
        self.__is_target_found = False

    def get_lvps_agent (self):
        return self.__lvps_sim_agent

    def get_model (self):
        return self.__model

    # attempts to estimate current position, as LVPS does
    def estimate_position (self, action_params):
        return self.__lvps_sim_agent.estimate_position()

    def is_occupied(self, pos):
        this_cell = self.model.grid.get_cell_list_contents([pos])
        return any(isinstance(agent, SearchAgent) for agent in this_cell)

    def adjust_randomly (self, action_params):
        return self.__lvps_sim_agent.adjust_randomly()

    def go (self, action_params):
        return self.__lvps_sim_agent.go(action_params=action_params)
    
    def go_forward (self, action_params):
        return self.__lvps_sim_agent.go_forward(action_params=action_params)

    def go_reverse (self, action_params):
        return self.__lvps_sim_agent.go_reverse(action_params=action_params)

    def go_forward_short (self,action_params):
        return self.__lvps_sim_agent.go_forward_short()

    def go_forward_medium (self,action_params):
        return self.__lvps_sim_agent.go_forward_medium()

    def go_forward_far (self,action_params):
        return self.__lvps_sim_agent.go_forward_far()

    def go_reverse_short (self,action_params):
        return self.__lvps_sim_agent.go_reverse_short()

    def go_reverse_medium (self,action_params):
        return self.__lvps_sim_agent.go_reverse_medium()

    def go_reverse_far (self,action_params):
        return self.__lvps_sim_agent.go_reverse_far()

    def rotate_left_small (self,action_params):
        return self.__lvps_sim_agent.rotate_left_small()

    def rotate_left_medium (self,action_params):
        return self.__lvps_sim_agent.rotate_left_medium()

    def rotate_left_big (self,action_params):
        return self.__lvps_sim_agent.rotate_left_big()

    def rotate_right_small (self,action_params):
        return self.__lvps_sim_agent.rotate_right_small()

    def rotate_right_medium (self,action_params):
        return self.__lvps_sim_agent.rotate_right_medium()

    def rotate_right_big (self,action_params):
        return self.__lvps_sim_agent.rotate_right_big()

    def adjust_randomly (self, action_params):
        return self.__lvps_sim_agent.adjust_randomly(action_params=action_params)

    def rotate(self, action_params):
        return self.__lvps_sim_agent.rotate(action_params=action_params)

    def photograph(self, action_params):
        return self.__lvps_sim_agent.photograph()

    def report_found(self, action_params):
        return self.__lvps_sim_agent.report_found()


    def notify_event (self, event_type):
        if event_type == 'found':
            logging.getLogger(__name__).info(f"System notified agent {self.unique_id} the search is over")
            self.__is_target_found = True
        else:
            logging.getLogger(__name__).info(f"event {event_type}")


    def look(self, action_params):
        return self.__lvps_sim_agent.look()
    
    def do_nothing (self, action_params):
        return self.__lvps_sim_agent.do_nothing()

    def step(self):
        if self.__is_target_found:
            # we are done
            return
        
        self.__step_count += 1
        action_map = {
            AgentActions.EstimatePosition : self.estimate_position,
            AgentActions.Go : self.go,
            AgentActions.Look : self.look,
            AgentActions.Rotate : self.rotate,
            AgentActions.Photograph : self.photograph,
            AgentActions.Nothing : self.do_nothing,
            AgentActions.ReportFound : self.report_found,
            AgentActions.GoForward : self.go_forward,
            AgentActions.GoReverse : self.go_reverse,
            AgentActions.AdjustRandomly : self.adjust_randomly,
            AgentActions.GoForwardShort : self.go_forward_short,
            AgentActions.GoForwardMedium : self.go_forward_medium,
            AgentActions.GoForwardFar : self.go_forward_far,
            AgentActions.GoReverseShort : self.go_reverse_short,
            AgentActions.GoReverseMedium : self.go_reverse_medium,
            AgentActions.GoReverseFar : self.go_reverse_far,
            AgentActions.RotateLeftSmall : self.rotate_left_small,
            AgentActions.RotateLeftMedium : self.rotate_left_medium,
            AgentActions.RotateLeftBig : self.rotate_left_big,
            AgentActions.RotateRightSmall : self.rotate_right_small,
            AgentActions.RotateRightMedium : self.rotate_right_medium,
            AgentActions.RotateRightBig : self.rotate_right_big
        }
        
        # check if current operation takes more steps
        if AgentActions.StepCost[self.__curr_action] > self.__step_count - self.__curr_action_start_time:
            #logging.getLogger(__name__).info("Action takes more time, waiting")
            pass
        else:
            next_action, next_config = self.__agent_strategy.get_next_action(self.__lvps_sim_agent, self.__curr_action, self.__curr_action_result, self.__step_count)

            # execute the action
            self.__curr_action = next_action
            self.__curr_action_start_time = self.__step_count
            self.__curr_action_result = action_map[next_action](next_config)

    def __get_distance(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx**2 + dy**2)    
    
class Target(mesa.Agent):
    def __init__(self, unique_id, pos, model):
        super().__init__(unique_id, model)
        logging.getLogger(__name__).info(f"Target added at {pos}")

    def step(self):
        pass # if target is allowed to move, this is where to do it

class Obstacle(mesa.Agent):
    def __init__(self, unique_id, pos, model):
        super().__init__(unique_id, model)
        #logging.getLogger(__name__).info(f"Obstacle added at {pos}")

    def step(self):
        pass

class Boundary(mesa.Agent):
    def __init__(self, unique_id, pos, model):
        super().__init__(unique_id, model)
        #logging.getLogger(__name__).info(f"Boundary added at {pos}")

    def step(self):
        pass
