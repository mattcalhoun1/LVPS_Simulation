from lvps.generators.field_map_generator import FieldMapGenerator
class RandomFieldMapGenerator (FieldMapGenerator):
    def __init__(self, min_width, max_width, num_obstacles, min_obstacle_size_pct, max_obstacle_size_pct):
        super.__init__()

    def generate_map (self):
        new_map = None
        # Create map
        with open('lvps/visual/resources/basement_v2.json', 'r') as mapin:
            json_map = json.loads(mapin.read())
            new_map = FieldMap(
                boundaries=json_map['boundaries'],
                shape=json_map['shape'],
                landmarks=json_map['landmarks'],
                obstacles=json_map['obstacles'],
                search=json_map['search'],
                near_boundaries=json_map['near_boundaries'],
                name='basement_v2'
            )

        return new_map