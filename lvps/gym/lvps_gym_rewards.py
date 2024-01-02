from lvps.strategies.agent_actions import AgentActions
import logging
import numpy as np

class RewardAmounts:
    StepCostMultiplier = 0.01 # to scale costs down , if desired

    OutOfBounds = -.1
    Collision = -.1
    InObstacle = -.1

    FalseReport = -.1

    SuccessfulLook = 10
    SuccessfulCloserLook = 10
    SuccessfulPhotoDistance = 30
    SuccessfulPhotograph = 50

    SomeAgentFoundTarget = 500
    ThisAgentFoundTarget = 500

    AllTargetsFound = 10000

class LvpsGymRewards:
    def __init__(self, agent):
        self.__agent = agent
        self.__target_find_counts = {}
        self.__target_closer_counts = {}
        self.__target_photo_distance_counts = {}
        self.__target_photo_counts = {}
        self.__target_last_dist = {}

    
    def calculate_reward (self, action_performed, action_result, target_found : bool, target_found_by_this_agent : bool, all_targets_found : bool, beg_nearest_unfound_target_id, beg_nearest_unfound_target_dist : float, end_nearest_unfound_target_id, end_nearest_unfound_target_dist : float, is_within_photo_distance : bool):

        if type(action_performed) is np.array or type(action_performed) is np.ndarray:
            action_performed = action_performed.max()

        ##### Penalties #####
        # every action costs a given number of steps
        reward = -1 * AgentActions.StepCost[action_performed] * RewardAmounts.StepCostMultiplier

        # if agent went out of bounds, extremely bad. depending on game settings, this ends the game
        if self.__agent.is_out_of_bounds():
            reward += RewardAmounts.OutOfBounds
        
        # if agent collided, bad
        if self.__agent.get_lvps_environment().is_too_close_to_other_agents(self.__agent.get_id()):
            reward += RewardAmounts.Collision

        # touching an obstacle is bad. depending on game settings, this ends the game
        if self.__agent.is_in_obstacle():
            reward += RewardAmounts.InObstacle

        # falsely reporting a target is bad
        if action_performed == AgentActions.ReportFound and action_result == False:
            reward += RewardAmounts.FalseReport

        ###### Rewards #######
        if action_performed == AgentActions.Look and action_result == True:
            # have we already seen this target before
            if end_nearest_unfound_target_id is not None and end_nearest_unfound_target_id not in self.__target_find_counts:
                reward += RewardAmounts.SuccessfulLook
                self.__target_find_counts[end_nearest_unfound_target_id] = 1
                self.__target_last_dist[end_nearest_unfound_target_id] = end_nearest_unfound_target_dist
                logging.getLogger(__name__).info(f"Agent rewarded for first time sighting of {end_nearest_unfound_target_id}")
            # have we closed the distance to the target in question, get rewarded for first time
            elif end_nearest_unfound_target_id is not None and end_nearest_unfound_target_id not in self.__target_closer_counts:
                if is_within_photo_distance == False and end_nearest_unfound_target_dist < self.__target_last_dist[end_nearest_unfound_target_id]:
                    self.__target_closer_counts[end_nearest_unfound_target_id] = 1
                    self.__target_last_dist[end_nearest_unfound_target_dist] = end_nearest_unfound_target_dist # also tracking in case we want to reward more distance closing at some point
                    logging.getLogger(__name__).info(f"Agent rewarded for closing distance to {end_nearest_unfound_target_id}")
                    reward += RewardAmounts.SuccessfulCloserLook
            
            # if we are within photo distance for the first time, reward that
            if is_within_photo_distance and end_nearest_unfound_target_id not in self.__target_photo_distance_counts:
                self.__target_photo_distance_counts[end_nearest_unfound_target_id] = 1
                reward += RewardAmounts.SuccessfulPhotoDistance
                logging.getLogger(__name__).info(f"Agent rewarded for getting within photo distance of {end_nearest_unfound_target_id}")


        if action_performed == AgentActions.Photograph and action_result == True:
            if end_nearest_unfound_target_id not in self.__target_photo_counts:
                logging.getLogger(__name__).info(f"Agent rewarded for photo of {end_nearest_unfound_target_id}")
                reward += RewardAmounts.SuccessfulPhotograph
                self.__target_photo_counts[end_nearest_unfound_target_id] = 1

        if target_found:
            logging.getLogger(__name__).info(f"Agent rewarded for target find (across all agents)")
            reward += RewardAmounts.SomeAgentFoundTarget
        
        if target_found_by_this_agent:
            logging.getLogger(__name__).info(f"Agent rewarded for target find (this agent found)")
            reward += RewardAmounts.ThisAgentFoundTarget

        if all_targets_found:
            logging.getLogger(__name__).info(f"Agent rewarded for all targets found")
            reward += RewardAmounts.AllTargetsFound

        return reward