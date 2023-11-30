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

    def get_scaled_coords (self, lvps_x, lvps_y):
        return self.__get_shifted_coords(*(self.__scale_coords_down(lvps_x,lvps_y)))
    
    def is_coord_visible (self, sim_perspective_x, sim_perspective_y, sim_target_x, sim_target_y, sight_range_sim):
        if self.__get_distance(sim_perspective_x, sim_perspective_y, sim_target_x, sim_target_y) <= sight_range_sim:
            #logging.getLogger(__name__).info("coord is within dist")
            lvps_p_x, lvps_p_y = self.get_lvps_coords(sim_perspective_x, sim_perspective_y)
            lvps_target_x, lvps_target_y = self.get_lvps_coords(sim_target_x, sim_target_y)

            blocked, obstacle_id = self.__field_map.is_path_blocked(lvps_p_x, lvps_p_y, lvps_target_x, lvps_target_y)
            return not blocked
        return False
    
    # checks whether given coord is blocked based on lvps coords
    def __is_lvps_coord_blocked (self, lvps_starting_x, lvps_starting_y, lvps_x, lvps_y):
        if lvps_starting_y == lvps_y and lvps_starting_x == lvps_x:
            return False

        #logging.getLogger(__name__).info("IN lvps check")
        bound_x_min, bound_y_min, bound_x_max, bound_y_max = self.__field_map.get_boundaries()
        if lvps_x <= bound_x_min or lvps_x >= bound_x_max or lvps_y <= bound_y_min or lvps_y >= bound_y_max:
            #logging.getLogger(__name__).info(f"{lvps_x},{lvps_y} is out of bounds")
            return True

        is_blocked, obstacle_id = self.__field_map.is_blocked(lvps_x, lvps_y)
        if is_blocked:
            #logging.getLogger(__name__).info(f"{lvps_x},{lvps_y} is blocked by {obstacle_id}")
            return True

        path_blocked, obstacle_id = self.__field_map.is_path_blocked (lvps_starting_x, lvps_starting_y, lvps_x, lvps_y)
        #if not path_blocked:
        #    logging.getLogger(__name__).info(f"lvps:LVPS Path from {lvps_starting_x},{lvps_starting_y} to {lvps_x},{lvps_y} is OPEN")
        #else:
        #    logging.getLogger(__name__).info(f"Lvps path from {lvps_starting_x},{lvps_starting_y} to {lvps_x},{lvps_y} is blockedQ!")

        return path_blocked
    
    def __is_sim_coord_blocked (self, sim_starting_x, sim_starting_y, sim_x, sim_y):
        if sim_starting_y == sim_y and sim_starting_x == sim_x:
            return False

        if self.is_in_bounds(sim_x, sim_y) == False or self.is_obstacle(sim_x, sim_y):
            return True
        
        # check map blockage
        lvps_start_x, lvps_start_y = self.get_lvps_coords(sim_starting_x, sim_starting_y)
        lvps_target_x, lvps_target_y = self.get_lvps_coords(sim_x, sim_y)

        path_blocked, obstacle_id = self.__field_map.is_path_blocked (lvps_start_x, lvps_start_y, lvps_target_x, lvps_target_y)
        #if not path_blocked:
        #    logging.getLogger(__name__).info(f"sim:LVPS Path from {lvps_start_x},{lvps_start_y} to {lvps_target_x},{lvps_target_y} is OPEN")
        return path_blocked

        
    def get_nearest_travelable_lvps_coords (self, starting_x, starting_y, target_x, target_y, max_dist):
        return self.__get_nearest_travelable_coords (starting_x, starting_y, target_x, target_y, max_dist, self.__is_lvps_coord_blocked)

    def get_nearest_travelable_sim_coords (self, starting_x, starting_y, target_x, target_y, max_dist):
        return self.__get_nearest_travelable_coords (starting_x, starting_y, target_x, target_y, max_dist, self.__is_sim_coord_blocked)


    def __get_nearest_travelable_coords (self, starting_x, starting_y, target_x, target_y, max_dist, block_check_method):
        # Get the point on the line between here and the target that is no farther than max dist
        # im sure there is a simple math calculation to do this, but i'm skipping that for now
        traveled = 0
        curr_x = starting_x
        curr_y = starting_y
        last_x = curr_x
        last_y = curr_y
        last_ok_x = curr_x
        last_ok_y = curr_y

        full_dist = self.__get_distance(curr_x, curr_y, target_x, target_y)
        #logging.getLogger(__name__).info(f"Want to go {full_dist} toward {target_x},{target_y}")

        # if the distance is near enough, go all the way
        #if full_dist <= max_dist:
        #    return target_x, target_y

        slope = (target_y - starting_y) / (target_x - starting_x)
        #logging.getLogger(__name__).info(f"tavbelable slope: {slope}")
        #logging.getLogger(__name__).info(f"Check1: {traveled < max_dist}")
        #logging.getLogger(__name__).info(f"Check2: {full_dist > self.__get_distance(starting_x, starting_y, curr_x, curr_y)}")
        #logging.getLogger(__name__).info(f"Check3: {block_check_method(starting_x, starting_y, curr_x, curr_y) == False}")

        while (traveled < max_dist and full_dist > self.__get_distance(starting_x, starting_y, curr_x, curr_y)):
            last_x = curr_x
            last_y = curr_y

            traveled += 1
            if abs(slope) < 1:
                curr_x += 1 if target_x > curr_x else -1
                curr_y += slope if target_x > curr_x else -1 * slope
            else:
                curr_y += slope if target_x > curr_x else -1 * slope
                curr_x += 1/slope if target_x > curr_x else -1 * 1/slope

            #logging.getLogger(__name__).info(f"Checking {curr_x},{curr_y}")
            if self.__get_distance(starting_x, starting_y, curr_x, curr_y) <= max_dist:
                if block_check_method(starting_x, starting_y, curr_x, curr_y) == False:
                    last_ok_x = curr_x
                    last_ok_y = curr_y

        #if traveled >= max_dist or block_check_method(starting_x, starting_y, curr_x, curr_y):
        #    logging.getLogger(__name__).info(f"returning nearest coords: {last_x},{last_y}")
        #    return last_x, last_y
        
        #logging.getLogger(__name__).info(f"returning nearest coords: {last_ok_x},{last_ok_y}")

        return last_ok_x, last_ok_y


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
