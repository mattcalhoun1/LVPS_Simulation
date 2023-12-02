import json
from field.field_map import FieldMap
from lvps.generators.field_map_generator import FieldMapGenerator

class StaticFieldMapGenerator(FieldMapGenerator):
    def __init__(self):
        super().__init__()

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

        # add some dead spots
        # dead spot in ne corner facing east
        new_map.add_dead_spot ('deadspot_ne', 75, 100, 100, 150, 33.0, 120.0)

        # dead spot sw corner facing all directions
        new_map.add_dead_spot ('deadspot_sw', -120.0, -75.0, -90.0, -20.0)

        return new_map