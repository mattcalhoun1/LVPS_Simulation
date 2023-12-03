import math
import logging
from lvps.strategies.agent_actions import AgentActions
from lvps.strategies.agent_strategy import AgentStrategy
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
        obstacle_bound = lvps_agent.is_in_obstacle()

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
            logging.getLogger(__name__).warning(f"Agent stuck in obstacle, going to random location")
            return AgentActions.GoToSafePlace, action_params

        # if we took a photo, check it and report any findings
        elif last_action == AgentActions.Photograph and last_action_result == True:
            # estimated position will be derived from info obtained from last photo
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
