import math
import logging
from .agent_actions import AgentActions
import numpy as np

class AgentStrategy:
    def __init__(self):
        pass

    def get_next_action (self, lvps_agent, last_action, last_action_result, step_count):
        logging.getLogger(__name__).error("AgentStrategy should be subclassed!")
        return AgentActions.Nothing, {}
