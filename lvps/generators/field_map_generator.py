import json
from field.field_map import FieldMap

class FieldMapGenerator:
    def __init__(self):
        pass

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