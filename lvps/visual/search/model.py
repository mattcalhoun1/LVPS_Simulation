import mesa
import json
import math
import logging

from .agents import SearchAgent, Target, Obstacle, Boundary
from .field_guide import FieldGuide
from lvps.generators.field_map_generator import FieldMapGenerator

class AutonomousSearch(mesa.Model):
    def __init__(self, width=120, height=120, num_robots=2, num_targets=1):
        """
        Create a new search episode

        Args:
            num_robots: Number of robots performing search
        """

        # Set parameters
        self.__map = None
        self.__field_guide = None
        self.width = width
        self.height = height
        self.num_robots = num_robots
        self.num_targets = num_targets
        self.__found = False

        self.schedule = mesa.time.RandomActivationByType(self)
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)
        self.datacollector = mesa.DataCollector(
            {"SearchAgent": lambda m: m.schedule.get_type_count(SearchAgent)}
        )

        guide = self.get_field_guide()

        agent_id = 0
        for _, (x, y) in self.grid.coord_iter():
            # if there's a barrier here, add it
            if not guide.is_in_bounds(x,y):
                boundary = Boundary(agent_id, (x,y), self)
                agent_id += 1
                self.grid.place_agent(boundary, (x, y))
            elif guide.is_obstacle(x,y):
                obstacle = Obstacle(agent_id, (x,y), self)
                agent_id += 1
                self.grid.place_agent(obstacle, (x, y))

        # Create agent:
        for i in range(self.num_robots):
            x,y = guide.get_random_traversable_coords()

            vision = self.random.randrange(1, 6) # change this
            sa = SearchAgent(agent_id, (x, y), self, guide)
            agent_id += 1
            self.grid.place_agent(sa, (x, y))
            self.schedule.add(sa)

        # create targets
        self.__targets = []
        for i in range(self.num_targets):
            x,y = guide.get_random_traversable_coords()
            target_agent = Target(agent_id, (x,y), self)
            agent_id += 1
            self.grid.place_agent(target_agent, (x, y))
            self.schedule.add(target_agent)

        self.running = True
        self.datacollector.collect(self)

    def get_field_guide (self):
        if self.__field_guide is None:
            self.__field_guide = FieldGuide(field_map=self.get_map(), simulator_height=self.height, simulator_width=self.width)
        
        return self.__field_guide

    # returns targets within sight range of the given position
    def get_visible_targets (self, x, y, sight_range):
        visible_targets = []
        for t in self.__targets:
            if self.__field_guide.is_coord_visible (sim_perspective_x = x, sim_perspective_y = y, sim_target_x = t.pos[0], sim_target_y = t.pos[1], sight_range_sim=sight_range):
                visible_targets.append(t)

        return visible_targets

    def get_distance(self, pos_1, pos_2):
        """Get the distance between two point

        Args:
            pos_1, pos_2: Coordinate tuples for both points.
        """
        x1, y1 = pos_1
        x2, y2 = pos_2
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx**2 + dy**2)

    def get_map (self):
        if self.__map is None:
            self.__map = FieldMapGenerator().generate_map()
        return self.__map

    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def report_target_found (self, agent_id, x, y):
        logging.getLogger(__name__).info(f"Agent {agent_id} reports finding the target at ({x},{y})")
        self.__found = True

    def run_model(self, step_count=200):
        logging.getLogger(__name__).info(f"Search Agents: {self.schedule.get_type_count(SearchAgent)}")

        for i in range(step_count):
            self.step()

            if self.__found:
                break

