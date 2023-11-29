import logging
import random
import math

# translates simulation coordinates to lvps coords and also does the reverse

class FieldGuide:
    def __init__(self, field_map, simulator_height, simulator_width):
        self.__field_map = field_map
        self.__simulator_height = simulator_height
        self.__simulator_width = simulator_width
        self.__map_scaler = .33 # 1 grid = 10" **2

        bound_x_min, bound_y_min, bound_x_max, bound_y_max = field_map.get_boundaries()
        bound_center_x = bound_x_min + ((bound_x_max - bound_x_min) / 2)
        bound_center_y = bound_y_min + ((bound_y_max - bound_y_min) / 2)
        scaled_down_center_x, scaled_down_center_y = self.__scale_coords_down(bound_center_x, bound_center_y)

        self.__shift_x = (self.__simulator_width / 2) - scaled_down_center_x
        self.__shift_y = (self.__simulator_height / 2) - scaled_down_center_y


    def get_map (self):
        return self.__field_map
    
    def scale_lvps_distance_to_sim (self, dist):
        return dist * self.__map_scaler
    
    def scale_sim_to_lvps_distance (self, dist):
        return dist / self.__map_scaler

    def get_lvps_coords (self, sim_x, sim_y):
        return self.__scale_coords_up(*self.__get_unshifted_coords(sim_x, sim_y))
    
    def get_sim_coords (self, lvps_x, lvps_y):
        return self.__get_shifted_coords(*(self.__scale_coords_down(lvps_x,lvps_y)))
    
    def is_coord_visible (self, sim_perspective_x, sim_perspective_y, sim_target_x, sim_target_y, sight_range_sim):
        if self.__get_distance(sim_perspective_x, sim_perspective_y, sim_target_x, sim_target_y) <= sight_range_sim:
            lvps_p_x, lvps_p_y = self.get_lvps_coords(sim_perspective_x, sim_perspective_y)
            lvps_target_x, lvps_target_y = self.get_lvps_coords(sim_target_x, sim_target_y)

            return not self.__field_map.is_path_blocked(lvps_p_x, lvps_p_y, lvps_target_x, lvps_target_y)
        return False

    def get_nearest_travelable_coords (self, sim_x, sim_y, target_x, target_y, max_dist):
        # Get the point on the line between here and the target that is no farther than max dist
        # im sure there is a simple math calculation to do this, but i'm skipping that for now
        traveled = 0
        curr_x = sim_x
        curr_y = sim_y
        last_x = curr_x
        last_y = curr_y

        last_dist = self.__get_distance(curr_x, curr_y, target_x, target_y)

        # if the distance is near enough, go all the way
        if last_dist <= max_dist:
            return target_x, target_y

        slope = (target_y - sim_y) / (target_x - sim_x)

        while (traveled < max_dist and last_dist >= self.__get_distance(curr_x, curr_y, target_x, target_y)):
            last_x = curr_x
            last_y = curr_y

            traveled += 1
            if abs(slope) < 1:
                curr_x += 1 if target_x > curr_x else -1
                curr_y += slope
            else:
                curr_y += slope
                curr_x += 1/slope

        if traveled >= max_dist:
            return last_x, last_y

        return curr_x, curr_y            


    def is_in_bounds (self, sim_x, sim_y):
        lvps_x,lvps_y = self.get_lvps_coords(sim_x, sim_y)
        bound_x_min, bound_y_min, bound_x_max, bound_y_max = self.__field_map.get_boundaries()
        if lvps_x <= bound_x_min or lvps_x >= bound_x_max or lvps_y <= bound_y_min or lvps_y >= bound_y_max:
            return False
        return True

    def is_obstacle (self, sim_x, sim_y):
        lvps_x,lvps_y = self.get_lvps_coords(sim_x, sim_y)
        is_blocked, obstacle_id = self.__field_map.is_blocked(lvps_x, lvps_y)
        return is_blocked

    def get_random_traversable_coords (self):
        max_attempts = 1000
        curr_attempt = 0

        while curr_attempt < max_attempts:
            curr_attempt += 1
            x = random.randrange(self.__simulator_width)
            y = random.randrange(self.__simulator_height)
            lvps_x, lvps_y = self.get_lvps_coords(x,y)

            if self.__field_map.is_in_bounds(lvps_x, lvps_y):
                blocked, obstacle_id = self.__field_map.is_blocked(lvps_x, lvps_y)
                if not blocked:
                    return x,y

        logging.getLogger(__name__).error("Unable to find traversable coords!")
        return None,None

    def __get_shifted_coords (self, x, y):
        return (x + self.__shift_x, y + self.__shift_y)

    def __get_unshifted_coords (self, x, y):
        return (x - self.__shift_x, y - self.__shift_y)

    def __scale_coords_up (self, x, y):
        scaled_x = x / self.__map_scaler
        scaled_y = y / self.__map_scaler

        scaled_x += (1 / self.__map_scaler) / 2
        scaled_y += (1 / self.__map_scaler) / 2

        #logging.getLogger(__name__).info(f"Simulation coord {(x,y)} scales up to LVPS {(scaled_x,scaled_y)}")

        return scaled_x, scaled_y

    def __scale_coords_down (self, x, y):
        scaled_x = x * self.__map_scaler
        scaled_y = y * self.__map_scaler

        # round or floor?
        scaled_x = round(scaled_x)
        scaled_y = round(scaled_y)

        #logging.getLogger(__name__).info(f"LVPS coord {(x,y)} scales down to {(scaled_x,scaled_y)}")

        return scaled_x, scaled_y    

    def __get_distance(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx**2 + dy**2)
