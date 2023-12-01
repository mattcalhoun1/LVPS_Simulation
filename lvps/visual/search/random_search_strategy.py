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
            # get back in bounds
            return SearchAgentActions.GoToSafePlace, action_params
        elif last_action == SearchAgentActions.ReportFound or (last_action == SearchAgentActions.Photograph and last_action_result == True):
            action_params['x'] = lvps_x
            action_params['y'] = lvps_y
            return SearchAgentActions.ReportFound, action_params
        elif last_action == SearchAgentActions.Look and last_action_result == True:
            # a target should be visible
            lvps_target_x, lvps_target_y = lvps_agent.get_nearest_visible_target_position()
            if lvps_target_x is not None:
                # the target was sighted, if it's within photo range, take a photo
                if self.__get_distance(lvps_x, lvps_y, lvps_target_x, lvps_target_y) <= lvps_agent.get_photo_distance():
                    return SearchAgentActions.Photograph, action_params
                else:
                    action_params['x'] = lvps_target_x
                    action_params['y'] = lvps_target_y
                    return SearchAgentActions.Go, action_params
        elif last_action == SearchAgentActions.EstimatePosition and last_action_result == True:
            return SearchAgentActions.Look, action_params
        #else:
        #    return SearchAgentActions.GoRandom, action_params
        #    # move randomly!
        #    #lvps_target_x, lvps_target_y = self.__get_random_coords ()

        # nothing was sighted, we have a position, choose a random direction to go
        return SearchAgentActions.GoRandom, action_params


    def go_random (self, action_params):
        logging.getLogger(__name__).info(f"Going in random direction")
        # find neighboring coords that are open and not in our recent history
        chosen_coords = None
        max_attempts = 10
        attempts = 0
        while (chosen_coords is None and attempts < max_attempts):
            attempts += 1
            # travel the sight distance
            desired_slope = np.random.choice([-4, -2, -1, -1.5, -.5, 0, .5, 1, 1.5, 2, 4])
            desired_direction = np.random.choice([-1, 1]) # x go left or right

            x_travel = desired_direction * self.get_sight_distance()
            far_x = self.__lvps_x + x_travel
            far_y = self.__lvps_y + x_travel * desired_slope
            closer_x, closer_y = self.__scaler.get_nearest_travelable_lvps_coords (self.__lvps_x, self.__lvps_y, far_x, far_y, self.get_sight_distance())

            # must be at least some distance from any of our past few moves
            backtracking = False
            for lh in self.__look_history if len(self.__look_history) < self.__look_history_size else self.__look_history[-1*self.__look_history_size:]:
                if self.__get_distance(closer_x, closer_y, lh[0], lh[1]) < (self.get_sight_distance() / 2):
                    backtracking = True

            # if we are not backtracking, or we're running out of directions to go            
            if backtracking == False or attempts >= (max_attempts - 1):
                sim_x, sim_y = self.__scaler.get_scaled_coords(self.__lvps_x, self.__lvps_y)
                target_sim_x, target_sim_y = self.__scaler.get_scaled_coords(closer_x, closer_y)

                if target_sim_x != sim_x and target_sim_y != sim_y:
                    final_x, final_y = self.__scaler.get_nearest_travelable_sim_coords (sim_x, sim_y, target_sim_x, target_sim_y, self.get_max_sim_travel_distance())

                    adjusted_x = self.__get_less_accurate(final_x, SearchAgentActions.Accuracy[SearchAgentActions.GoRandom])
                    adjusted_y = self.__get_less_accurate(final_y, SearchAgentActions.Accuracy[SearchAgentActions.GoRandom])

                    chosen_coords = (round(adjusted_x), round(adjusted_y))

                    logging.getLogger(__name__).info(f"Random traveling from sim coords: {sim_x},{sim_y} to {final_x},{final_y}, randomly adjusted to {round(adjusted_x)},{round(adjusted_y)}")
        
        if chosen_coords is not None:
            self.__sim_travel(chosen_coords[0], chosen_coords[1])

            self.__lvps_x = None
            self.__lvps_y = None

            return self.__does_event_happen(SearchAgentActions.GoRandom)
        
        return False


    def __get_distance(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx**2 + dy**2)