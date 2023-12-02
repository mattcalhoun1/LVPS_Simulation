import math
import logging
from lvps.simulation.agent_actions import AgentActions
from lvps.simulation.agent_strategy import AgentStrategy
import numpy as np
from trig.trig import BasicTrigCalc

# this is a truly random strategy. it is aweful, but calls all the same methods that will be used by training

class RandomSearchStrategy(AgentStrategy):
    def __init__(self):
        super().__init__()
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

        # if we don't know where we are, need to figure that out
        if (lvps_x is None or lvps_y is None) and last_action != AgentActions.EstimatePosition:
            # see if the last action was to estimate (and failed)
            return AgentActions.EstimatePosition, action_params
        elif (lvps_x is not None or lvps_y is not None) and last_action == AgentActions.EstimatePosition:
            # look after every step
            return AgentActions.Look, action_params
        
        # if we are stuck or out of bounds, reset toward the center
        if lvps_agent.is_out_of_bounds():
            logging.getLogger(__name__).warning(f"Agent is out of bounds, going to random location")
            return AgentActions.GoToSafePlace, action_params
        elif obstacle_bound:
            logging.getLogger(__name__).warning(f"Agent stuck in obstacle {obstacle_id}, going to random location")
            return AgentActions.GoToSafePlace, action_params

        # if we took a photo, check it and report any findings
        elif last_action == AgentActions.Photograph and last_action_result == True:
            lvps_target_x, lvps_target_y, lvps_target_heading = lvps_agent.get_nearest_photographable_target_position()

            est_x, est_y = self.__trig_calc.get_coords_for_zeronorth_angle_and_distance (
                heading=lvps_target_heading,
                x = lvps_x,
                y = lvps_y,
                distance = lvps_agent.get_photo_distance()/2,
                is_forward = True)
            #est_x, est_y = self.__get_estimated_object_x_y (lvps_target_heading, lvps_x, lvps_y, lvps_agent.get_photo_distance()/2, 0)
            logging.getLogger(__name__).info(f"Real target at {lvps_target_x},{lvps_target_y} ... estimated: {est_x},{est_y}")

            action_params['x'] = est_x
            action_params['y'] = est_y
            action_params['heading'] = lvps_target_heading
            action_params['distance'] = lvps_agent.get_photo_distance()

            return AgentActions.ReportFound, action_params
        
        # otherwise, do something random
        action_space = [
            AgentActions.GoForwardShort,
            AgentActions.GoForwardMedium,
            AgentActions.GoForwardFar,
            AgentActions.GoReverseShort,
            AgentActions.GoReverseMedium,
            AgentActions.GoReverseFar,
            AgentActions.RotateLeftSmall,
            AgentActions.RotateLeftMedium,
            AgentActions.RotateLeftBig,
            AgentActions.RotateRightSmall,
            AgentActions.RotateRightMedium,
            AgentActions.RotateRightBig,
            AgentActions.Look,
            AgentActions.Photograph,
            AgentActions.Nothing
        ]

        return np.random.choice(action_space), action_params
