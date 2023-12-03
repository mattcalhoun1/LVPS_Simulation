from lvps.strategies.agent_actions import AgentActions

class RewardAmounts:
    StepCostMultiplier = 1 # to scale costs down , if desired

    OutOfBounds = -100
    Collision = -100
    InObstacle = -100

    FalseReport = -10

    SomeAgentFoundTarget = 500
    ThisAgentFoundTarget = 500

    AllTargetsFound = 10000

class LvpsGymRewards:
    def __init__(self, agent):
        self.__agent = agent
    
    def calculate_reward (self, action_performed, action_result, target_found : bool, target_found_by_this_agent : bool, all_targets_found : bool):

        ##### Penalties #####
        # every action costs a given number of steps
        reward = -1 * AgentActions.StepCost[action_performed] * RewardAmounts.StepCostMultiplier

        # if agent went out of bounds, extremely bad
        if self.__agent.is_out_of_bounds():
            reward += RewardAmounts.OutOfBounds
        
        # if agent collided, bad
        if self.__agent.get_lvps_environment().is_too_close_to_other_agents(self.__agent.get_id()):
            reward += RewardAmounts.Collision

        # touching an obstacle is bad
        if self.__agent.is_in_obstacle():
            reward += RewardAmounts.InObstacle

        # falsely reporting a target is bad
        if action_performed == AgentActions.ReportFound and action_result == False:
            reward += RewardAmounts.FalseReport

        ###### Rewards #######
        if target_found:
            reward += RewardAmounts.SomeAgentFoundTarget
        
        if target_found_by_this_agent:
            reward += RewardAmounts.ThisAgentFoundTarget

        if all_targets_found:
            reward += RewardAmounts.AllTargetsFound

        return reward