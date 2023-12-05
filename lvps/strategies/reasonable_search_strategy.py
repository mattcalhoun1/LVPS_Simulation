import math
import logging
from lvps.strategies.agent_actions import AgentActions
from lvps.strategies.agent_strategy import AgentStrategy
from lvps.simulation.simulated_agent import SimulatedAgent
from trig.trig import BasicTrigCalc
import numpy as np

class ReasonableSearchStrategy(AgentStrategy):
    def __init__(self, render_field = True):
        super().__init__()
        self.__consecutive_position_fails = 0
        self.__max_position_fails = 2
        self.__trig_calc = BasicTrigCalc()
        self.__render_field = render_field

    def get_next_action (self, lvps_agent : SimulatedAgent, last_action, last_action_result, step_count):
        action_params = {
            'agent':lvps_agent
        }

        lvps_x, lvps_y, lvps_heading, lvps_confidence = lvps_agent.get_last_coords_and_heading()
        obstacle_bound = lvps_agent.is_in_obstacle()

        # render the field as an image
        if self.__render_field:
            lvps_agent.get_field_renderer().save_field_image(
                f'/tmp/lvpssim/agent_{lvps_agent.get_id()}_step_{step_count}.png',
                add_game_state=True,
                agent_id=lvps_agent.get_id(),
                other_agents_visible=True,
                width_inches=4,
                height_inches=4,
                dpi=100)

        # if we just tried to get position and it failed, adjust the vehicle a little and try again
        if last_action == AgentActions.EstimatePosition:
            if last_action_result == False:
                self.__consecutive_position_fails += 1
                if self.__consecutive_position_fails > self.__max_position_fails:
                    return AgentActions.AdjustRandomly, action_params
            else:
                self.__consecutive_position_fails = 0

        # if we don't know where we are, need to figure that out
        if lvps_x is None or lvps_y is None:
            return AgentActions.EstimatePosition, action_params

        # if we went out of bounds, go to our safe space (wherever that is)
        if lvps_agent.is_out_of_bounds():
            logging.getLogger(__name__).warning(f"Agent is out of bounds, going to random location")
            return self.__queue_go_to_safe_place (lvps_agent, action_params)

        # if we are stuck to an obstacle, try to get out
        elif obstacle_bound:
            logging.getLogger(__name__).warning(f"Agent stuck in obstacle, going to random location")
            return self.__queue_go_to_safe_place (lvps_agent, action_params)
        
        elif last_action == AgentActions.ReportFound:
            # need to move to a random location so we dont get stuck here
            # even if the report fails. if we found the same item again, the report fails
            return self.__queue_go_random(lvps_agent=lvps_agent, action_params=action_params)
        elif last_action == AgentActions.Photograph and last_action_result == True:
            # position info will be derived from last photo search
            return AgentActions.ReportFound, action_params
        elif (last_action == AgentActions.Look and last_action_result == True):
            # a target should be visible
            lvps_target_x, lvps_target_y, lvps_target_heading = lvps_agent.get_nearest_visible_target_position()
            if lvps_target_x is not None:
                # the target was sighted, if it's within photo range, take a photo
                photo_x, photo_y, photo_heading = lvps_agent.get_nearest_photographable_target_position()
                #if self.__get_distance(lvps_x, lvps_y, lvps_target_x, lvps_target_y) <= lvps_agent.get_photo_distance():
                if photo_x is not None:
                    return AgentActions.Photograph, action_params
                else:
                    action_params['x'] = lvps_target_x
                    action_params['y'] = lvps_target_y
                    action_params['distance_percent'] = 0.25 # go 25% toward it
                    return AgentActions.Go, action_params
        elif last_action == AgentActions.EstimatePosition and last_action_result == True:
            return AgentActions.Look, action_params

        #else:
        #    return SearchAgentActions.GoRandom, action_params
        #    # move randomly!
        #    #lvps_target_x, lvps_target_y = self.__get_random_coords ()

        # nothing was sighted, we have a position, choose a random direction to go
        return self.__queue_go_random(lvps_agent=lvps_agent, action_params=action_params)

    def __get_distance(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx**2 + dy**2)
    
    def __queue_go_to_safe_place (self, lvps_agent : SimulatedAgent, action_params):
        logging.getLogger(__name__).info(f"Going to safe place")
        safe_x, safe_y = lvps_agent.get_field_renderer().get_map_scaler().get_random_traversable_coords()
        action_params['x'] = safe_x
        action_params['y'] = safe_y
        return AgentActions.Go, action_params

    def __queue_go_random (self, lvps_agent : SimulatedAgent, action_params):
        logging.getLogger(__name__).info(f"Going in random direction")
        # find neighboring coords that are open and not in our recent history
        chosen_coords = None
        max_attempts = 10
        attempts = 0
        min_travel_dist = 5.0 # dont bother moving if it's less than this

        lvps_x, lvps_y, lvps_heading, lvps_confidence = lvps_agent.get_last_coords_and_heading()
        chosen_x = None
        chosen_y = None

        while (chosen_x is None and attempts < max_attempts):
            attempts += 1
            # travel the sight distance
            desired_slope = np.random.choice([-4, -2, -1, -1.5, -.5, 0, .5, 1, 1.5, 2, 4])
            desired_direction = np.random.choice([-1, 1]) # x go left or right

            x_travel = desired_direction * lvps_agent.get_sight_distance()
            far_x = lvps_x + x_travel
            far_y = lvps_y + x_travel * desired_slope
            closer_x, closer_y = lvps_agent.get_field_renderer().get_map_scaler().get_nearest_travelable_lvps_coords (lvps_x, lvps_y, far_x, far_y, lvps_agent.get_sight_distance())

            # must be at least some distance from any of our past few moves
            backtracking = False
            look_history = lvps_agent.get_look_history()
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
            logging.getLogger(__name__).debug(f"Random traveling from coords: {lvps_x},{lvps_y} to {chosen_x},{chosen_y}")
            return AgentActions.Go, action_params
       
        return AgentActions.Nothing, action_params