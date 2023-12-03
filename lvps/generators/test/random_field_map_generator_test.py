import unittest
from lvps.generators.random_field_map_generator import RandomFieldMapGenerator
import logging
import json

class RandomMapGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        return super().setUp()

    def test_square (self):
        generator = RandomFieldMapGenerator(
            min_width=250,
            max_width=500,
            min_height=250,
            max_height=500,
            min_obstacles=3,
            max_obstacles=10,
            min_obstacle_size_pct=0.01,
            max_obstacle_size_pct=0.25,
            min_deadspots=2,
            max_deadspots=10,
            min_deadspot_size_pct = 0.01,
            max_deadspot_size_pct=0.05
        )

        logging.getLogger(__name__).info(f"Generated: {json.dumps(generator.generate_map_dict())}")