import math
import random
import mesa
import logging
from .field_guide import FieldGuide
from .search_agent_actions import SearchAgentActions
from .action_strategy import ActionStrategy
import numpy as np

class SearchAgent(mesa.Agent):
    def __init__(self, unique_id, pos, model, guide : FieldGuide=None):
        super().__init__(unique_id, model)
        self.__guide = guide
        self.__model = model
        self.__unique_id = unique_id
        self.pos = pos # position within the simulation

        self.__step_count = 0
        self.__lvps_x = None
        self.__lvps_y = None
        self.__total_distance_traveled = 0

        self.__moore = True # can see in diagonal directions, not just up/down/left/right
        self.__sight_distance = 150 # how far can see, lvps distance
        self.__photo_distance = 50 # how close needs to be to get a good photo, lvps distance
        self.__max_travel_distance = self.__guide.get_map().get_width() * 0.25 # how far can travel in one leg
        logging.getLogger(__name__).info(f"Robot added at {pos}")

        self.__curr_action = SearchAgentActions.Nothing
        self.__curr_action_start_time = 0
        self.__curr_action_result = True

        self.__look_history_size = 10 # remember this many past looks
        self.__look_history = [] # where it has already looked

        self.__action_strategy = ActionStrategy()

    # returns fixed size look history, intended to be fed into a policy model of some sort
    def __get_look_history (self):
        look_memory = []
        for h in self.__look_history:
            look_memory.append(h)
        
        while len(look_memory) < self.__look_history_size:
            look_memory.append((0,0))
        return look_memory
    
    def get_total_distance_traveled(self):
        return self.__total_distance_traveled

    def get_max_travel_distance (self):
        return self.__max_travel_distance
    
    def get_max_sim_travel_distance (self):
        return self.__guide.scale_lvps_distance_to_sim(self.__max_travel_distance)

    def get_sim_sight_distance(self):
        return self.__guide.scale_lvps_distance_to_sim(self.__sight_distance)

    def get_sight_distance (self):
        return self.__sight_distance

    def get_sim_photo_distance(self):
        return self.__guide.scale_lvps_distance_to_sim(self.__photo_distance)

    def get_photo_distance (self):
        return self.__photo_distance

    def get_model (self):
        return self.__model

    # attempts to estimate current position, as LVPS does
    def estimate_position (self, action_params):
        logging.getLogger(__name__).info("Estimate position")
        success = False

        # get actual position
        if self.__does_event_happen(SearchAgentActions.SuccessRate[SearchAgentActions.EstimatePosition]):
            success = True
            lvps_x, lvps_y = self.__guide.get_lvps_coords(self.pos[0], self.pos[1])

            # corrupt the position a little bit
            self.__lvps_x = self.__get_less_accurate(lvps_x, SearchAgentActions.Accuracy[SearchAgentActions.EstimatePosition])
            self.__lvps_y = self.__get_less_accurate(lvps_y, SearchAgentActions.Accuracy[SearchAgentActions.EstimatePosition])
        else:
            self.__lvps_x = None
            self.__lvps_y = None
        return success

    def is_occupied(self, pos):
        this_cell = self.model.grid.get_cell_list_contents([pos])
        return any(isinstance(agent, SearchAgent) for agent in this_cell)
    
    def go_random (self, action_params):
        logging.getLogger(__name__).info(f"Going in random direction")
        # find neighboring coords that are open and not in our recent history
        chosen_coords = None
        desired_dist = self.get_sight_distance()
        max_attempts = 10
        attempts = 0
        while (chosen_coords is None and attempts < max_attempts):
            attempts += 1
            # travel the sight distance
            desired_slope = np.random.choice([-4, -2, -1, -1.5, -.5, 0, .5, 1, 1.5, 2, 4])
            desired_direction = np.random.choice([-1, 1]) # x go left or right

            x_travel = desired_direction * self.get_sight_distance()
            far_x = self.__lvps_x + x_travel
            far_y = self.__lvps_y + x_travel * desired_slope
            closer_x, closer_y = self.__guide.get_nearest_travelable_lvps_coords (self.__lvps_x, self.__lvps_y, far_x, far_y, self.get_sight_distance())

            # must be at least some distance from any of our past few moves
            backtracking = False
            for lh in self.__look_history:
                if self.__get_distance(closer_x, closer_y, lh[0], lh[1]) < (self.get_sight_distance() / 2):
                    backtracking = True

            # if we are not backtracking, or we're running out of directions to go            
            if backtracking == False or attempts >= (max_attempts - 1):
                sim_x, sim_y = self.__guide.get_sim_coords(self.__lvps_x, self.__lvps_y)
                target_sim_x, target_sim_y = self.__guide.get_sim_coords(closer_x, closer_y)

                if target_sim_x != sim_x and target_sim_y != sim_y:
                    final_x, final_y = self.__guide.get_nearest_travelable_sim_coords (sim_x, sim_y, target_sim_x, target_sim_y, self.get_max_sim_travel_distance())

                    adjusted_x = self.__get_less_accurate(final_x, SearchAgentActions.Accuracy[SearchAgentActions.GoRandom])
                    adjusted_y = self.__get_less_accurate(final_y, SearchAgentActions.Accuracy[SearchAgentActions.GoRandom])

                    chosen_coords = (round(adjusted_x), round(adjusted_y))

                    logging.getLogger(__name__).info(f"Random traveling from sim coords: {sim_x},{sim_y} to {final_x},{final_y}, randomly adjusted to {round(adjusted_x)},{round(adjusted_y)}")
        
        if chosen_coords is not None:
            self.__sim_travel(chosen_coords[0], chosen_coords[1])

            self.__lvps_x = None
            self.__lvps_y = None

            return self.__does_event_happen(SearchAgentActions.GoRandom)
        
        return False

    def __sim_travel (self, next_sim_x, next_sim_y):
        self.__total_distance_traveled += self.__guide.scale_sim_to_lvps_distance(self.__get_distance(self.pos[0], self.pos[1], next_sim_x, next_sim_y))

        self.model.grid.move_agent(self, (round(next_sim_x), round(next_sim_y)))

    def go (self, action_params):
        logging.getLogger(__name__).info(f"Going toward {action_params['x']},{action_params['y']}")

        target_x = action_params['x']
        target_y = action_params['y']

        adjusted_x = self.__get_less_accurate(target_x, SearchAgentActions.Accuracy[SearchAgentActions.Go])
        adjusted_y = self.__get_less_accurate(target_y, SearchAgentActions.Accuracy[SearchAgentActions.Go])

        sim_x, sim_y = self.__guide.get_sim_coords(adjusted_x, adjusted_y)

        next_sim_x, next_sim_y = self.__guide.get_nearest_travelable_sim_coords (sim_x, sim_y, target_x, target_y, self.get_max_sim_travel_distance())

        self.__sim_travel(next_sim_x, next_sim_y)        
        self.__lvps_x = None
        self.__lvps_y = None

        return self.__does_event_happen(SearchAgentActions.Go)

    def rotate(self, action_params):
        logging.getLogger(__name__).info("Rotating")

    def photograph(self, action_params):
        logging.getLogger(__name__).info("Photographing")
        lvps_target_x, lvps_target_y = self.get_nearest_photographable_target_position(
            self.__lvps_x, 
            self.__lvps_y,
            action_params['guide'],
            action_params['agent'].get_model())

        # if a target is visible, this was a success
        success = lvps_target_x is not None and lvps_target_y is not None

        success = success and self.__does_event_happen(SearchAgentActions.SuccessRate[SearchAgentActions.Photograph])
        logging.getLogger(__name__).info(f"Photographing {'was' if success else 'was not'} successful")
        return success

    def report_found(self, action_params):
        logging.getLogger(__name__).info(f"Reporting found at: {action_params['x']},{action_params['y']}")
        self.__model.report_target_found (agent_id = self.unique_id, x=action_params['x'], y=action_params['y'])

        return self.__does_event_happen(SearchAgentActions.SuccessRate[SearchAgentActions.ReportFound])


    def look(self, action_params):
        # if we are within distance of the search target and it's not in a blind spot, or obscured, find it (random chance)
        logging.getLogger(__name__).info("Looking")
        self.__look_history.insert(0, (self.__lvps_x, self.__lvps_y))

        lvps_target_x, lvps_target_y = self.get_nearest_visible_target_position(
            self.__lvps_x, 
            self.__lvps_y,
            action_params['guide'],
            action_params['agent'].get_model())

        # if a target is visible, this was a success
        success = lvps_target_x is not None and lvps_target_y is not None
        if success:
            logging.getLogger(__name__).info("Found a Target visually!")

        # drop old history
        if len(self.__look_history) > self.__look_history_size:
            del self.__look_history[len(self.__look_history) - 1]

        return success and self.__does_event_happen(SearchAgentActions.SuccessRate[SearchAgentActions.Look])
    
    def do_nothing (self, action_params):
        logging.getLogger(__name__).info("Doing nothing")
        return True

    def step(self):
        self.__step_count += 1
        action_map = {
            SearchAgentActions.EstimatePosition : self.estimate_position,
            SearchAgentActions.Go : self.go,
            SearchAgentActions.Look : self.look,
            SearchAgentActions.Rotate : self.rotate,
            SearchAgentActions.Photograph : self.photograph,
            SearchAgentActions.GoRandom : self.go_random,
            SearchAgentActions.Nothing : self.do_nothing,
            SearchAgentActions.ReportFound : self.report_found
        }
        
        # check if current operation takes more steps
        if SearchAgentActions.StepCost[self.__curr_action] > self.__step_count - self.__curr_action_start_time:
            logging.getLogger(__name__).info("Action takes more time, waiting")
        else:
            next_action, next_config = self.__action_strategy.get_next_action(self, self.__lvps_x, self.__lvps_y, self.__guide, self.__curr_action, self.__curr_action_result)

            # execute the action
            self.__curr_action = next_action
            self.__curr_action_start_time = self.__step_count
            self.__curr_action_result = action_map[next_action](next_config)
       
    def __get_less_accurate(self, good_value, accuracy):
        # choose a random amount to mess with the calculation
        tolerance = 1 - accuracy
        adjust_amount = random.randrange(int(-1 * 1000 * tolerance), int(1000 * tolerance)) / 1000
        return good_value + (good_value * adjust_amount)


    # returns true if a "random" event should occur, based on the given occurance rate
    def __does_event_happen (self, occurance_rate):
        return random.randrange(0,100) <= occurance_rate * 100
    
    def __get_distance(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx**2 + dy**2)    

    # returns lvps coords for the nearest visible target (if any)
    def get_nearest_visible_target_position (self, lvps_x, lvps_y, guide, model):
        sim_x, sim_y = guide.get_sim_coords(lvps_x, lvps_y)

        # returns the distance and direciotn of nearest target to help us decide if the robot will be able to see it
        min_dist = None
        closest = None

        logging.getLogger(__name__).info(f"Checking for visible targets within sim dist: {self.get_sim_sight_distance()}")
        for t in model.get_visible_targets(sim_x, sim_y, self.get_sim_sight_distance()):
            dist = self.__get_distance(sim_x, sim_y, t.pos[0], t.pos[1])
            if min_dist is None or dist < min_dist:
                closest = t

        if closest is None:
            return None,None
        
        return guide.get_lvps_coords(closest.pos[0], closest.pos[1])
        
    # returns lvps coords for the nearest photographable target (if any)
    def get_nearest_photographable_target_position (self, lvps_x, lvps_y, guide, model):
        sim_x, sim_y = guide.get_sim_coords(lvps_x, lvps_y)

        # returns the distance and direciotn of nearest target to help us decide if the robot will be able to see it
        min_dist = None
        closest = None

        for t in model.get_visible_targets(sim_x, sim_y, self.get_sim_photo_distance()):
            dist = self.__get_distance(sim_x, sim_y, t.pos[0], t.pos[1])
            if min_dist is None or dist < min_dist:
                closest = t

        if closest is None:
            return None,None
        return guide.get_lvps_coords(closest.pos[0], closest.pos[1])
    
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
