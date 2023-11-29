import math
import logging
from .search_agent_actions import SearchAgentActions
import numpy as np

class ActionStrategy:
    def __init__(self):
        pass

    def get_next_action (self, agent, lvps_x, lvps_y, guide, last_action, last_action_result):
        action_params = {
            'agent':agent,
            'guide':guide,
        }

        # if we don't know where we are, need to figure that out
        if lvps_x is None or lvps_y is None:
            return SearchAgentActions.EstimatePosition, action_params
        elif last_action == SearchAgentActions.ReportFound or (last_action == SearchAgentActions.Photograph and last_action_result == True):
            action_params['x'] = lvps_x
            action_params['y'] = lvps_y
            return SearchAgentActions.ReportFound, action_params
        elif last_action == SearchAgentActions.Look and last_action_result == True:
            # a target should be visible
            lvps_target_x, lvps_target_y = agent.get_nearest_visible_target_position(lvps_x, lvps_y, guide, agent.get_model())
            if lvps_target_x is not None:
                # the target was sighted, if it's within photo range, take a photo
                if self.__get_distance(lvps_x, lvps_y, lvps_target_x, lvps_target_y) <= agent.get_photo_distance():
                    return SearchAgentActions.Photograph, action_params
                else:
                    action_params['x'] = lvps_target_x
                    action_params['y'] = lvps_target_y
                    return SearchAgentActions.Go, action_params
        elif last_action == SearchAgentActions.EstimatePosition and last_action_result == True:
            return SearchAgentActions.Look, action_params

        # nothing was sighted, we have a position, choose a random direction to go
        return SearchAgentActions.GoRandom, action_params


    def __get_distance(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx**2 + dy**2)