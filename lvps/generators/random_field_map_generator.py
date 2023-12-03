from lvps.generators.field_map_generator import FieldMapGenerator
from field.field_map_persistence import FieldMapPersistence
import numpy as np

class RandomFieldMapGenerator (FieldMapGenerator):
    def __init__(self, min_width, max_width, min_height, max_height, min_obstacles, max_obstacles, min_obstacle_size_pct, max_obstacle_size_pct, min_deadspots, max_deadspots, min_deadspot_size_pct, max_deadspot_size_pct):
        super().__init__()

        self.__min_width = min_width
        self.__max_width = max_width

        self.__min_height = min_height
        self.__max_height = max_height

        self.__center_x_min = -200.0
        self.__center_x_max = 200.0
        self.__center_y_min = -200.0
        self.__center_y_max = 200.0

        self.__min_obstacles = min_obstacles
        self.__max_obstacles = max_obstacles
        self.__min_obstacle_size_pct = min_obstacle_size_pct
        self.__max_obstacle_size_pct = max_obstacle_size_pct

        self.__min_deadspots = min_deadspots
        self.__max_deadspots = max_deadspots
        self.__min_deadspot_size_pct = min_deadspot_size_pct
        self.__max_deadspot_size_pct = max_deadspot_size_pct


    def generate_map (self):
        map_dict = self.generate_map_dict()

        return FieldMapPersistence().load_map_from_dict(map_dict)

    def generate_map_dict (self):
        new_map_dict = {
            'shape':'rectangle',
        }

        self.__add_boundaries(new_map_dict)
        self.__add_landmarks(new_map_dict)
        self.__add_obstacles(new_map_dict)
        self.__add_dead_spots(new_map_dict)

        return new_map_dict

    def __add_boundaries (self, map_json):
        # center the map in a random location
        center_x = np.random.choice(np.linspace(self.__center_x_min, self.__center_x_max))
        center_y = np.random.choice(np.linspace(self.__center_y_min, self.__center_y_max))

        width = np.random.choice(np.linspace(self.__min_width, self.__max_width))
        height = np.random.choice(np.linspace(self.__min_height, self.__max_height))

        map_json['boundaries'] = {
            'xmin':center_x - (width / 2),
            'xmax':center_x + (width / 2),
            'ymin':center_y - (height / 2),
            'ymax':center_y + (height / 2),
        }

    def __add_landmarks (self, map_json):
        map_json['landmarks'] = {}

    def __add_dead_spots (self, map_json):
        map_json['dead_spots'] = {}
        num_deadspots = np.random.randint(self.__min_deadspots, self.__max_deadspots+1)
        boundaries = map_json['boundaries']

        min_ds_width = self.__min_deadspot_size_pct * (boundaries['xmax'] - boundaries['xmin'])
        max_ds_width = self.__max_deadspot_size_pct * (boundaries['xmax'] - boundaries['xmin'])
        min_ds_height = self.__min_deadspot_size_pct * (boundaries['ymax'] - boundaries['ymin'])
        max_ds_height = self.__max_deadspot_size_pct * (boundaries['ymax'] - boundaries['ymin'])

        for o in range(num_deadspots):
            map_json['dead_spots'][f'ds_{o}'] = self.__get_coords_random_size_shape(
                min_height=min_ds_height,
                max_height=max_ds_height,
                min_width=min_ds_width,
                max_width=max_ds_width,
                boundaries=boundaries,
                used_spaces=map_json['dead_spots'],
                overlap_allowed=False
            )

    def __add_obstacles (self, map_json):
        map_json['obstacles'] = {}
        num_obstacles = np.random.randint(self.__min_obstacles, self.__max_obstacles+1)
        boundaries = map_json['boundaries']

        min_obstacle_width = self.__min_obstacle_size_pct * (boundaries['xmax'] - boundaries['xmin'])
        max_obstacle_width = self.__max_obstacle_size_pct * (boundaries['xmax'] - boundaries['xmin'])
        min_obstacle_height = self.__min_obstacle_size_pct * (boundaries['ymax'] - boundaries['ymin'])
        max_obstacle_height = self.__max_obstacle_size_pct * (boundaries['ymax'] - boundaries['ymin'])

        for o in range(num_obstacles):
            map_json['obstacles'][f'obstacle_{o}'] = self.__get_coords_random_size_shape(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                min_width=min_obstacle_width,
                max_width=max_obstacle_width,
                boundaries=boundaries,
                used_spaces=map_json['obstacles'],
                overlap_allowed=False
            )
            
    
    def __get_coords_random_size_shape (self, min_height : float, max_height : float, min_width : float, max_width : float, boundaries : tuple, used_spaces : list, overlap_allowed : bool):
        height = np.random.choice(np.linspace(min_height, max_height))
        width = np.random.choice(np.linspace(min_width, max_width))
        obj_min_x = None
        obj_min_y = None
        obj_max_x = None
        obj_max_y = None

        # randomly select a different center x until the given width fits in an area not already taken
        found_fit = False
        while found_fit == False:
            center_x = np.random.choice(np.linspace(boundaries['xmin'], boundaries['xmax']))
            center_y = np.random.choice(np.linspace(boundaries['ymin'], boundaries['ymax']))
            obj_min_x, obj_min_y, obj_max_x, obj_max_y = self.__get_centered_coords(center_x=center_x, center_y=center_y, height=height, width=width)

            max_moves = 10
            move_count = 0
            potential_x_moves = np.linspace((boundaries['xmax'] - boundaries['xmin']) / -2, (boundaries['xmax'] - boundaries['xmin']) / 2)
            potential_y_moves = np.linspace((boundaries['ymax'] - boundaries['ymin']) / -2, (boundaries['ymax'] - boundaries['ymin']) / 2)
            while found_fit == False and move_count < max_moves:
                found_fit = self.__object_fits(
                    obj_min_x,
                    obj_min_y,
                    obj_max_x,
                    obj_max_y,
                    boundaries=boundaries,
                    used_spaces=used_spaces,
                    overlap_allowed=overlap_allowed)
                
                if found_fit == False:
                    # move the center of the object in a random direction
                    center_x = np.random.choice(potential_x_moves)
                    center_y = np.random.choice(potential_y_moves)                    
                    obj_min_x, obj_min_y, obj_max_x, obj_max_y = self.__get_centered_coords(center_x=center_x, center_y=center_y, height=height, width=width)
                move_count += 1
        
        return {'xmin':obj_min_x, 'ymin': obj_min_y, 'xmax': obj_max_x, 'ymax': obj_max_y}

    def __get_centered_coords (self, center_x, center_y, height, width):
            obj_min_x = center_x - (width/2)
            obj_max_x = center_x + (width/2)
            obj_min_y = center_y - (height/2)
            obj_max_y = center_y + (height/2)

            return obj_min_x, obj_min_y, obj_max_x, obj_max_y

    
    def __object_fits (self, min_x : float, min_y : float, max_x : float, max_y : float, boundaries : tuple, used_spaces : list, overlap_allowed : bool):
        if min_x >= boundaries['xmin'] and max_x <= boundaries['xmax'] and min_y >= boundaries['ymin'] and max_y <= boundaries['ymax']:
            # check for overlap with any used space
            if not overlap_allowed:
                for sp in used_spaces:
                    sp_coords = used_spaces[sp]
                    # see if the x coords overlap
                    if (sp_coords['xmin'] > min_x and sp_coords['xmin'] < max_x) or (sp_coords['xmax'] > min_x and sp_coords['xmax'] < max_x):
                        # see if y coords overlap
                        if (sp_coords['ymin'] > min_y and sp_coords['ymin'] < max_y) or (sp_coords['ymax'] > min_y and sp_coords['ymax'] < max_y):
                            return False
            
            # either all used spaces have been checked or overlap is allowed
            return True
        else:
            return True

