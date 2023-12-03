import json
from field.field_map import FieldMap
from lvps.generators.field_map_generator import FieldMapGenerator
from field.field_map_persistence import FieldMapPersistence

class StaticFieldMapGenerator(FieldMapGenerator):
    def __init__(self):
        super().__init__()

    def generate_map (self):
        static_map = FieldMapPersistence().load_map('lvps/visual/resources/basement_v2.json')

        # add some dead spots
        # dead spot in ne corner facing east
        static_map.add_dead_spot ('deadspot_ne', 75, 100, 100, 150, 33.0, 120.0)

        # dead spot sw corner facing all directions
        static_map.add_dead_spot ('deadspot_sw', -120.0, -75.0, -90.0, -20.0)

        return static_map