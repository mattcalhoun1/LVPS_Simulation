import math
import random
import mesa
import logging
#from .field_guide import FieldGuide
from lvps.strategies.agent_actions import AgentActions
from .agent_types import AgentTypes
import numpy as np

from field.field_renderer import FieldRenderer
from field.field_map_persistence import FieldMapPersistence
from field.field_scaler import FieldScaler
from .lvps_sim_environment import LvpsSimEnvironment
from trig.trig import BasicTrigCalc

# represents a simulation for a single agent's perspective

class SimulatedAgent:
    def __init__(self, agent_id, agent_type, field_renderer : FieldRenderer, lvps_env : LvpsSimEnvironment, initial_x : float = None, initial_y : float = None, initial_heading : float = None, initial_confidence = None):
        self.__field_renderer = field_renderer
        self.__agent_id = agent_id
        self.__lvps_env = lvps_env
        self.__agent_type = agent_type

        self.__lvps_x = initial_x
        self.__lvps_y = initial_y
        self.__lvps_heading = initial_heading
        self.__lvps_confidence = initial_confidence
        self.__total_distance_traveled = 0

        # move these to configs
        self.__sight_distance = 45 # how far can see, lvps distance
        self.__photo_distance = 14 # how close needs to be to get a good photo, lvps distance
        self.__relative_search_begin = -150.0
        self.__relative_search_end = 150.0
        self.__max_travel_distance = self.__lvps_env.get_map().get_width() * 0.25 # how far can travel in one leg

        self.__far_distance_pct = 0.4
        self.__med_distance_pct = 0.2
        self.__short_distance_pct = 0.05

        self.__degrees_adjust_small = 6
        self.__degrees_adjust_medium = 33
        self.__degrees_adjust_big = 45
        
        self.__last_photo_x = None
        self.__last_photo_y = None
        self.__last_photo_heading = None

        self.__trig_calc = BasicTrigCalc()

        if self.__lvps_x is None:
            logging.getLogger(__name__).info(f"Agent {agent_id} added to LVPS simulation with no awareness of its current position.")
        else:
            logging.getLogger(__name__).info(f"Agent {agent_id} added to LVPS simulation, known to be at {self.__lvps_x}, {self.__lvps_y}.")

        self.__look_history = [] # where it has already looked

        self.__position_history = []

    def get_look_history (self):
        return self.__look_history

    def get_position_history (self):
        return self.__position_history

    def get_relative_search_begin (self):
        return self.__relative_search_begin
    
    def get_relative_search_end(self):
        return self.__relative_search_end

    def get_id (self):
        return self.__agent_id

    def get_total_distance_traveled(self):
        return self.__total_distance_traveled

    def get_max_travel_distance (self):
        return self.__max_travel_distance
    
    def get_sight_distance (self):
        return self.__sight_distance

    def get_photo_distance (self):
        return self.__photo_distance

    def get_lvps_environment (self):
        return self.__lvps_env
    
    def get_field_renderer (self):
        return self.__field_renderer

    def get_last_coords_and_heading (self):
        return self.__lvps_x, self.__lvps_y, self.__lvps_heading, self.__lvps_confidence

    # attempts to estimate current position, as LVPS does
    def estimate_position (self):
        logging.getLogger(__name__).debug(f"Agent {self.__agent_id} Estimate position")
        success = False

        lvps_x, lvps_y, heading, confidence = self.__lvps_env.estimate_agent_position(self.__agent_id)
        # get actual position
        if lvps_x is not None:
            self.__lvps_x = lvps_x
            self.__lvps_y = lvps_y
            self.__lvps_heading = heading
            self.__lvps_confidence = confidence

            success = True
            self.__position_history.append((self.__lvps_x, self.__lvps_y, self.__lvps_heading, self.__lvps_confidence))
            self.__update_agent_rendering()
        else:
            self.__lvps_x = None
            self.__lvps_y = None
            self.__lvps_heading = heading
            self.__lvps_confidence = confidence

        return success
    
    # updates the shared (or not shared) agents' rendering of the field with new information
    # this can be called any time with no side effects
    # it SHOULD be called any time the agent receives new information (new position, search hit, etc)
    def __update_agent_rendering (self):
        self.__field_renderer.update_agent_state(self.__agent_id, self.__position_history, [* reversed(self.__look_history)])

    def is_out_of_bounds (self):
        if self.__lvps_x is not None and self.__lvps_y is not None:
            return not self.__lvps_env.get_map().is_in_bounds(self.__lvps_x, self.__lvps_y, self.get_path_width())
        return False

    def is_in_obstacle (self):
        if self.__lvps_x is not None and self.__lvps_y is not None:
            blocked, obstacle_id = self.__lvps_env.get_map().is_blocked(self.__lvps_x, self.__lvps_y, self.get_path_width())
            return blocked
        return False


    def go_forward (self, action_params):
        success = self.__lvps_env.go_forward(self.__agent_id, action_params['distance'])
        self.__update_agent_rendering()

        self.__lvps_x = None
        self.__lvps_y = None
        self.__lvps_confidence = None
        self.__lvps_heading = None

        return success

    def go_reverse (self, action_params):
        success = self.__lvps_env.go_reverse(self.__agent_id, action_params['distance'])
        self.__update_agent_rendering()

        self.__lvps_x = None
        self.__lvps_y = None
        self.__lvps_confidence = None
        self.__lvps_heading = None

        return success

    def adjust_randomly (self):
        logging.getLogger(__name__).debug(f"Agent {self.__agent_id} Adjusting randomly")
        success = True

        move_methods = [
            self.__lvps_env.go_forward,
            self.__lvps_env.go_reverse,
        ]

        # If agent can strafe, prefer that
        if AgentTypes.SupportsStrafe[self.__agent_type]:
            move_methods.append(self.__lvps_env.strafe_left)
            move_methods.append(self.__lvps_env.strafe_left)
            move_methods.append(self.__lvps_env.strafe_right)
            move_methods.append(self.__lvps_env.strafe_right)

        selected_move = np.random.choice(move_methods)
        success = selected_move(self.__agent_id, self.get_short_distance())

        # choose a random rotation
        rotations = [-45.0, -33.0, -15.0, 0, 15.0, 33.0, 45.0]
        success = success or self.__lvps_env.rotate(self.__agent_id, np.random.choice(rotations))

        self.__lvps_x = None
        self.__lvps_y = None
        self.__lvps_confidence = None
        self.__lvps_heading = None

        self.__update_agent_rendering()

        return success            

    def get_path_width (self):
        return AgentTypes.PathWidth[self.__agent_type]

    # driving methods, to be utilized by learning algorithm.
    # these are discrete, rather than providing continuous params
    def go_forward_short (self):
        return self.go_forward({'distance':self.get_short_distance()})

    def go_forward_medium (self):
        return self.go_forward({'distance':self.get_medium_distance()})

    def go_forward_far (self):
        return self.go_forward({'distance':self.get_far_distance()})

    def go_reverse_short (self):
        return self.go_reverse({'distance':self.get_short_distance()})

    def go_reverse_medium (self):
        return self.go_reverse({'distance':self.get_medium_distance()})

    def go_reverse_far (self):
        return self.go_reverse({'distance':self.get_far_distance()})

    def rotate_left_small (self):
        return self.rotate({'degrees':self.__degrees_adjust_small * -1})

    def rotate_left_medium (self):
        return self.rotate({'degrees':self.__degrees_adjust_medium * -1})

    def rotate_left_big (self):
        return self.rotate({'degrees':self.__degrees_adjust_big * -1})

    def rotate_right_small (self):
        return self.rotate({'degrees':self.__degrees_adjust_small})

    def rotate_right_medium (self):
        return self.rotate({'degrees':self.__degrees_adjust_medium})

    def rotate_right_big (self):
        return self.rotate({'degrees':self.__degrees_adjust_big})


    def get_short_distance (self):
        return min(self.__lvps_env.get_map().get_width(),self.__lvps_env.get_map().get_length()) * self.__short_distance_pct

    def get_medium_distance (self):
        return min(self.__lvps_env.get_map().get_width(),self.__lvps_env.get_map().get_length()) * self.__med_distance_pct

    def get_far_distance (self):
        return min(self.__lvps_env.get_map().get_width(),self.__lvps_env.get_map().get_length()) * self.__far_distance_pct
        
    def __has_recent_position (self):
        return self.__lvps_x is not None and self.__lvps_y is not None

    def go (self, action_params):

        if not self.__has_recent_position():
            return False

        logging.getLogger(__name__).debug(f"Going toward {action_params['x']},{action_params['y']}")

        target_x = action_params['x']
        target_y = action_params['y']

        if 'distance_percent' in action_params and action_params['distance_percent'] != 1.0:
            # get a fraction of the distance
            full_dist = self.__get_distance(self.__lvps_x, self.__lvps_y, target_x, target_y)
            target_x, target_y = self.__field_renderer.get_map_scaler().get_nearest_travelable_lvps_coords(
                self.__lvps_x, 
                self.__lvps_y, 
                target_x, target_y,
                max_dist=full_dist * action_params['distance_percent'])
            logging.getLogger(__name__).debug(f"Shortened desired new position from {action_params['x']},{action_params['y']} to new target of {target_x},{target_y}")

        start_x = self.__lvps_x
        start_y = self.__lvps_y

        # clear any coords we currently have
        self.__lvps_x = None
        self.__lvps_y = None
        self.__lvps_confidence = None
        self.__lvps_heading = None

        #try to go
        success = self.__lvps_env.go(self.__agent_id, target_x=target_x, target_y=target_y)

        if success and start_x is not None:
            self.__total_distance_traveled += self.__get_distance(start_x, start_y, target_x, target_y)

        return success
    
    def rotate(self, action_params):
        success = self.__lvps_env.rotate(self.__agent_id, action_params['degrees'])
        self.__update_agent_rendering()

        self.__lvps_x = None
        self.__lvps_y = None
        self.__lvps_confidence = None
        self.__lvps_heading = None

        return success


    def photograph(self):
        if not self.__has_recent_position():
            return False

        logging.getLogger(__name__).info(f"Agent {self.__agent_id} photographing")

        lvps_target_x, lvps_target_y, lvps_target_heading = self.get_nearest_photographable_target_position()
        #self.__look_history.insert(0, (self.__lvps_x, self.__lvps_y, self.__lvps_heading, self.__relative_search_begin, self.__relative_search_end, self.get_photo_distance()))
        self.__update_agent_rendering()

        # if a target is visible, this was a success
        success = lvps_target_x is not None and lvps_target_y is not None

        success = success and self.__does_event_happen(AgentActions.SuccessRate[AgentActions.Photograph])
        logging.getLogger(__name__).info(f"~~~~~~~~ Photographing {'was' if success else 'was not'} successful ~~~~~~~~~~~~~~")


        if success:
            self.__last_photo_x = self.__lvps_x
            self.__last_photo_y = self.__lvps_y
            self.__last_photo_heading = self.__lvps_heading

        return success

    def report_found(self):
        if not self.__has_recent_position():
            return False

        # if we've moved since the photograph, we can't report
        if self.__last_photo_x != self.__lvps_x or self.__last_photo_y != self.__lvps_y or self.__last_photo_heading != self.__lvps_heading:
            return False


        # this is sort of cheating, as we are using the known coords in this case. In reality, we would be estimated the 
        # position, not calculating it.
        lvps_target_x, lvps_target_y, lvps_target_heading = self.get_nearest_photographable_target_position()

        est_x, est_y = self.__trig_calc.get_coords_for_zeronorth_angle_and_distance (
            heading=lvps_target_heading,
            x = self.__lvps_x,
            y = self.__lvps_y,
            distance = self.__get_distance(self.__lvps_x, self.__lvps_y, lvps_target_x, lvps_target_y),
            is_forward = True)


        # if the report turns out to be false, there will be a penalty. If it's true, there will be a big reward.
        logging.getLogger(__name__).info(f"Reporting found at: {est_x},{est_y}")
        if self.__does_event_happen(AgentActions.SuccessRate[AgentActions.ReportFound]):
            return self.__lvps_env.report_target_found (agent_id = self.__agent_id, x=est_x, y=est_y)
        return False

    def look(self):
        if not self.__has_recent_position():
            return False

        # if we are within distance of the search target and it's not in a blind spot, or obscured, find it (random chance)
        logging.getLogger(__name__).debug(f"Agent {self.__agent_id} Looking (facing {self.__lvps_heading})")
        self.__look_history.insert(0, (self.__lvps_x, self.__lvps_y, self.__lvps_heading, self.__relative_search_begin, self.__relative_search_end, self.get_sight_distance()))
        self.__update_agent_rendering()

        lvps_target_x, lvps_target_y, lvps_target_heading = self.get_nearest_visible_target_position()

        # if a target is visible, this was a success
        success = lvps_target_x is not None and lvps_target_y is not None
        if success:
            logging.getLogger(__name__).info(f"Agent {self.__agent_id} Found a Target visually!")

        return success and self.__does_event_happen(AgentActions.SuccessRate[AgentActions.Look])
    
    def do_nothing (self):
        logging.getLogger(__name__).debug("Doing nothing")
        return True
       
    # returns true if a "random" event should occur, based on the given occurance rate
    def __does_event_happen (self, occurance_rate):
        return random.randrange(0,100) <= occurance_rate * 100
    
    def __get_distance(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx**2 + dy**2)    

    # returns lvps coords for the nearest visible target (if any)
    def get_nearest_visible_target_position (self):
        # returns the distance and direciotn of nearest target to help us decide if the robot will be able to see it
        min_dist = None
        closest = None
        closest_heading = None

        logging.getLogger(__name__).debug(f"Checking for visible targets within sim dist: {self.get_sight_distance()}")
        vis_targets, vis_headings = self.__lvps_env.get_visible_targets(self.__agent_id, self.get_sight_distance())
        for i,t in enumerate(vis_targets):
            dist = self.__get_distance(self.__lvps_x, self.__lvps_y, t['x'], t['y'])
            if min_dist is None or dist < min_dist:
                closest = t
                closest_heading = vis_headings[i]

        if closest is None:
            return None,None,None
        
        return closest['x'],closest['y'], closest_heading
        

    # returns lvps coords for the nearest visible target (if any)
    def get_nearest_unfound_target_distance (self):
        # returns the distance and direciotn of nearest target to help us decide if the robot will be able to see it
        min_dist = None
        closest = None
        closest_heading = None

        logging.getLogger(__name__).debug(f"Checking for visible unfound targets within sim dist: {self.get_sight_distance()}")
        vis_targets, vis_headings = self.__lvps_env.get_visible_targets(self.__agent_id, self.get_sight_distance())
        for i,t in enumerate(vis_targets):
            if self.__lvps_env.is_target_found(self.__agent_id, t['x'], t['y']) == False:
                dist = self.__get_distance(self.__lvps_x, self.__lvps_y, t['x'], t['y'])
                if min_dist is None or dist < min_dist:
                    closest = t
                    closest_heading = vis_headings[i]

        if closest is None:
            return None,None,None
        
        return closest, min_dist, closest_heading
        
    # returns lvps coords for the nearest photographable target (if any)
    def get_nearest_photographable_target_position (self):
        # returns the distance and direciotn of nearest target to help us decide if the robot will be able to see it
        min_dist = None
        closest = None
        closest_heading = None

        vis_targets, vis_headings = self.__lvps_env.get_visible_targets(self.__agent_id, self.get_photo_distance())
        for i,t in enumerate(vis_targets):
            dist = self.__get_distance(self.__lvps_x, self.__lvps_y, t['x'], t['y'])
            if min_dist is None or dist < min_dist:
                closest = t
                closest_heading = vis_headings[i]


        if closest is None:
            return None,None,None
        return closest['x'], closest['y'], closest_heading
