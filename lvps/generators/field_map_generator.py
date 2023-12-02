import json
from field.field_map import FieldMap
import logging

class FieldMapGenerator:
    def __init__(self):
        pass

    def generate_map (self):
        logging.getLogger(__name__).fatal("FieldMapGenerator must be subclassed!")