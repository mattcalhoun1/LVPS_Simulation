import math
import logging
from .search_agent_actions import SearchAgentActions
from lvps.sim.agent_strategy import AgentStrategy
import numpy as np

class RandomSearchStrategy(AgentStrategy):
    def __init__(self):
        super().__init__()

    def get_next_action (self, lvps_agent, last_action, last_action_result, step_count):
        action_params = {
            'agent':lvps_agent
        }

        lvps_x, lvps_y, lvps_heading, lvps_confidence = lvps_agent.get_last_coords_and_heading()
        obstacle_bound = False
        obstacle_id = None

        if lvps_x is not None:
            obstacle_bound, obstacle_id = lvps_agent.get_lvps_environment().get_map().is_blocked(lvps_x, lvps_y)

        lvps_agent.get_field_renderer().save_field_image(
            f'/tmp/lvpssim/agent_{lvps_agent.get_id()}_step_{step_count}.png',
            add_game_state=True,
            agent_id=lvps_agent.get_id(),
            other_agents_visible=True)

        # if we don't know where we are, need to figure that out
        if lvps_x is None or lvps_y is None:
            return SearchAgentActions.EstimatePosition, action_params
        if lvps_agent.is_out_of_bounds():
            logging.getLogger(__name__).warning(f"Agent is out of bounds, going to random location")
            return SearchAgentActions.GoToSafePlace, action_params
        elif obstacle_bound:
            logging.getLogger(__name__).warning(f"Agent stuck in obstacle {obstacle_id}, going to random location")
            return SearchAgentActions.GoToSafePlace, action_params
        elif last_action == SearchAgentActions.ReportFound:
            # need to move to a random location so we dont get stuck here
            # even if the report fails. if we found the same item again, the report fails
            return SearchAgentActions.GoRandom, action_params
        elif last_action == SearchAgentActions.Photograph and last_action_result == True:
            lvps_target_x, lvps_target_y, lvps_target_heading = lvps_agent.get_nearest_photographable_target_position()

            # get estimated x, y..sort of cheating by using known x,y iwth our est x,y. in reality, we'd be using camera angles and est dist
            est_x, est_y = self.__get_estimated_object_x_y (lvps_target_heading, lvps_x, lvps_y, lvps_agent.get_photo_distance(), 0)

            logging.getLogger(__name__).info(f"Real target at {lvps_target_x},{lvps_target_y} ... estimated: {est_x},{est_y}")

            action_params['x'] = est_x
            action_params['y'] = est_y
            action_params['heading'] = lvps_target_heading
            action_params['distance'] = lvps_agent.get_photo_distance()

            return SearchAgentActions.ReportFound, action_params
        elif (last_action == SearchAgentActions.Look and last_action_result == True):
            # a target should be visible
            lvps_target_x, lvps_target_y, lvps_target_heading = lvps_agent.get_nearest_visible_target_position()
            if lvps_target_x is not None:
                # the target was sighted, if it's within photo range, take a photo
                if self.__get_distance(lvps_x, lvps_y, lvps_target_x, lvps_target_y) <= lvps_agent.get_photo_distance():
                    return SearchAgentActions.Photograph, action_params
                else:
                    action_params['x'] = lvps_target_x
                    action_params['y'] = lvps_target_y
                    action_params['distance_percent'] = 0.25 # go 25% toward it
                    return SearchAgentActions.Go, action_params
        elif last_action == SearchAgentActions.EstimatePosition and last_action_result == True:
            return SearchAgentActions.Look, action_params
        #else:
        #    return SearchAgentActions.GoRandom, action_params
        #    # move randomly!
        #    #lvps_target_x, lvps_target_y = self.__get_random_coords ()

        # nothing was sighted, we have a position, choose a random direction to go
        return SearchAgentActions.GoRandom, action_params

    def __get_estimated_object_x_y (self, heading, x, y, obj_dist, obj_degrees):
        # rotate degrees so zero is east and 180 is west
        #x = r X cos( θ )
        #y = r X sin( θ )
        cartesian_angle_degrees = 180 - (obj_degrees - heading)
        if cartesian_angle_degrees < 0:
            cartesian_angle_degrees += 360

        logging.getLogger(__name__).info(f"Finding object position using vehicle heading: {heading}, x: {x}, y:{y}, cartesian coord angle: {cartesian_angle_degrees}")

        est_x = x + obj_dist * math.cos(math.radians(cartesian_angle_degrees))
        est_y = y + obj_dist * math.sin(math.radians(cartesian_angle_degrees))
        return est_x, est_y

    def __get_distance(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx**2 + dy**2)