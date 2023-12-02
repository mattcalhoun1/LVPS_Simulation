import math
import logging
from lvps.simulation.agent_actions import AgentActions
from lvps.simulation.agent_strategy import AgentStrategy
from trig.trig import BasicTrigCalc
import numpy as np

class ReasonableSearchStrategy(AgentStrategy):
    def __init__(self):
        super().__init__()
        self.__consecutive_position_fails = 0
        self.__max_position_fails = 2
        self.__trig_calc = BasicTrigCalc()

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

        if last_action == AgentActions.EstimatePosition:
            if last_action_result == False:
                self.__consecutive_position_fails += 1
                if self.__consecutive_position_fails > self.__max_position_fails:
                    return AgentActions.AdjustRandomly, action_params
            else:
                # reset the counter and let actions proceed
                self.__consecutive_position_fails = 0

        # if we don't know where we are, need to figure that out
        if lvps_x is None or lvps_y is None:
            # see if the last action was to estimate (and failed)
            return AgentActions.EstimatePosition, action_params
        if lvps_agent.is_out_of_bounds():
            logging.getLogger(__name__).warning(f"Agent is out of bounds, going to random location")
            return AgentActions.GoToSafePlace, action_params
        elif obstacle_bound:
            logging.getLogger(__name__).warning(f"Agent stuck in obstacle {obstacle_id}, going to random location")
            return AgentActions.GoToSafePlace, action_params
        elif last_action == AgentActions.ReportFound:
            # need to move to a random location so we dont get stuck here
            # even if the report fails. if we found the same item again, the report fails
            return AgentActions.GoRandom, action_params
        elif last_action == AgentActions.Photograph and last_action_result == True:
            lvps_target_x, lvps_target_y, lvps_target_heading = lvps_agent.get_nearest_photographable_target_position()

            # get estimated x, y..sort of cheating by using known x,y iwth our est x,y. in reality, we'd be using camera angles and est dist
            est_x, est_y = self.__trig_calc.get_coords_for_zeronorth_angle_and_distance (
                heading=lvps_target_heading,
                x = lvps_x,
                y = lvps_y,
                distance = lvps_agent.get_photo_distance()/2,
                is_forward = True)

            logging.getLogger(__name__).info(f"Real target at {lvps_target_x},{lvps_target_y} ... estimated: {est_x},{est_y}")

            action_params['x'] = est_x
            action_params['y'] = est_y
            action_params['heading'] = lvps_target_heading
            action_params['distance'] = lvps_agent.get_photo_distance()

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
        return AgentActions.GoRandom, action_params

    def __get_distance(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx**2 + dy**2)