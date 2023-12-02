import math
import random
import mesa
import logging
#from .field_guide import FieldGuide
import numpy as np
from lvps.simulation.simulated_agent import SimulatedAgent
from lvps.simulation.agent_strategy import AgentStrategy
from lvps.simulation.agent_actions import AgentActions

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

        self.__moore = True # can see in diagonal directions, not just up/down/left/right

        self.__curr_action = AgentActions.Nothing
        self.__curr_action_start_time = 0
        self.__curr_action_result = True

        self.__look_history_size = 10 # remember this many past looks
        self.__look_history = [] # where it has already looked

        self.__position_history = []

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
        return self.__lvps_sim_agent.adjust_randomly(action_params)

    def go_random (self, action_params):
        logging.getLogger(__name__).info(f"Going in random direction")
        # find neighboring coords that are open and not in our recent history
        chosen_coords = None
        max_attempts = 10
        attempts = 0
        min_travel_dist = 5.0 # dont bother moving if it's less than this

        lvps_x, lvps_y, lvps_heading, lvps_confidence = self.__lvps_sim_agent.get_last_coords_and_heading()
        chosen_x = None
        chosen_y = None

        while (chosen_x is None and attempts < max_attempts):
            attempts += 1
            # travel the sight distance
            desired_slope = np.random.choice([-4, -2, -1, -1.5, -.5, 0, .5, 1, 1.5, 2, 4])
            desired_direction = np.random.choice([-1, 1]) # x go left or right

            x_travel = desired_direction * self.__lvps_sim_agent.get_sight_distance()
            far_x = lvps_x + x_travel
            far_y = lvps_y + x_travel * desired_slope
            closer_x, closer_y = self.__scaler.get_nearest_travelable_lvps_coords (lvps_x, lvps_y, far_x, far_y, self.__lvps_sim_agent.get_sight_distance())

            # must be at least some distance from any of our past few moves
            backtracking = False
            look_history = self.__lvps_sim_agent.get_look_history()
            if len(look_history) > 1:
                recent_history = look_history[len(look_history) - 2] # next to last. We probably alraedy looked from current coord
                recent_x = recent_history[0]
                recent_y = recent_history[1] 
                if self.__get_distance(closer_x, closer_y, recent_x, recent_y) < self.__get_distance(lvps_x, lvps_y, recent_x, recent_y):
                    backtracking = True

            # if we are not backtracking, or we're running out of directions to go            
            if backtracking == False or attempts >= (max_attempts - 1):
                if self.__get_distance(lvps_x, lvps_y, closer_x, closer_y) >= min_travel_dist:
                    chosen_x = closer_x
                    chosen_y = closer_y
                #else:
                #    logging.getLogger(__name__).info(f"Distance from {lvps_x},{lvps_y} to {closer_x},{closer_y} is not far enough")
                #    #    final_x, final_y = self.__scaler.get_nearest_travelable_sim_coords (sim_x, sim_y, target_sim_x, target_sim_y, self.get_max_sim_travel_distance())
                #    #    chosen_coords = (round(final_x), round(final_y))
        
        if chosen_x is not None:
            action_params['x'] = chosen_x
            action_params['y'] = chosen_y
            logging.getLogger(__name__).info(f"Random traveling from coords: {lvps_x},{lvps_y} to {chosen_x},{chosen_y}")
            return self.__lvps_sim_agent.go(action_params=action_params)
       
        return False

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

    def go_to_safe_place (self, action_params):
        safe_x, safe_y = self.__scaler.get_random_traversable_coords()
        action_params['x'] = safe_x
        action_params['y'] = safe_y
        return self.go(action_params=action_params)

    def rotate(self, action_params):
        return self.__lvps_sim_agent.rotate(action_params=action_params)

    def photograph(self, action_params):
        return self.__lvps_sim_agent.photograph()

    def report_found(self, action_params):
        return self.__lvps_sim_agent.report_found(action_params=action_params)


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
            AgentActions.GoRandom : self.go_random,
            AgentActions.Nothing : self.do_nothing,
            AgentActions.ReportFound : self.report_found,
            AgentActions.GoToSafePlace : self.go_to_safe_place,
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
            logging.getLogger(__name__).info("Action takes more time, waiting")
        else:
            next_action, next_config = self.__agent_strategy.get_next_action(self.__lvps_sim_agent, self.__curr_action, self.__curr_action_result, self.__step_count)

            # execute the action
            self.__curr_action = next_action
            self.__curr_action_start_time = self.__step_count
            self.__curr_action_result = action_map[next_action](next_config)

    # returns true if a "random" event should occur, based on the given occurance rate
    def __does_event_happen (self, occurance_rate):
        return random.randrange(0,100) <= occurance_rate * 100
    
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
