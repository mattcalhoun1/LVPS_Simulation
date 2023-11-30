import mesa
import json
import math
import logging

from .agents import SearchAgent, Target, Obstacle, Boundary
from field.field_scaler import FieldScaler
from lvps.generators.field_map_generator import FieldMapGenerator
from field.field_renderer import FieldRenderer

class AutonomousSearch(mesa.Model):
    num_robots=1

    def __init__(self, width=120, height=120, num_robots=1, num_targets=1):
        """
        Create a new search episode

        Args:
            num_robots: Number of robots performing search
        """

        # Set parameters
        self.__map = None
        self.__field_scaler = None
        self.__image_scaler = None
        self.width = width
        self.height = height
        self.num_robots = num_robots
        self.num_targets = num_targets
        self.__found = False

        self.schedule = mesa.time.RandomActivationByType(self)
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)
        self.datacollector = mesa.DataCollector({
            "TotalDistance": lambda m: m.get_total_distance_traveled()
            },
        )

        sim_scaler = self.get_field_sim_scaler()
        field_renderer = FieldRenderer(field_map = sim_scaler.get_map(), map_scaler=self.get_field_image_scaler(sim_scaler.get_map()))

        agent_id = 0
        for _, (x, y) in self.grid.coord_iter():
            # if there's a barrier here, add it
            if not sim_scaler.is_in_bounds(x,y):
                boundary = Boundary(agent_id, (x,y), self)
                agent_id += 1
                self.grid.place_agent(boundary, (x, y))
            elif sim_scaler.is_obstacle(x,y):
                obstacle = Obstacle(agent_id, (x,y), self)
                agent_id += 1
                self.grid.place_agent(obstacle, (x, y))

        # Create agent:
        self.__search_agents = []
        for i in range(self.num_robots):
            x,y = sim_scaler.get_random_traversable_coords()

            vision = self.random.randrange(1, 6) # change this
            sa = SearchAgent(agent_id, (x, y), self, sim_scaler, field_renderer=field_renderer)
            agent_id += 1
            self.grid.place_agent(sa, (x, y))
            self.schedule.add(sa)
            self.__search_agents.append(sa)

        # create targets
        self.__targets = []
        for i in range(self.num_targets):
            x,y = sim_scaler.get_random_traversable_coords()
            target_agent = Target(agent_id, (x,y), self)
            agent_id += 1
            self.grid.place_agent(target_agent, (x, y))
            self.schedule.add(target_agent)
            self.__targets.append(target_agent)

        self.running = True
        self.datacollector.collect(self)

    def get_total_distance_traveled (self):
        total = 0
        for a in self.__search_agents:
            total += a.get_total_distance_traveled()
        return total

    def get_field_sim_scaler (self):
        if self.__field_scaler is None:
            self.__field_scaler = FieldScaler(field_map=self.get_map(), scaled_height=self.height, scaled_width=self.width, scale_factor=0.33)
        
        return self.__field_scaler

    def get_field_image_scaler (self, field_map):
        if self.__image_scaler is None:
            rendered_height = 640
            rendered_width = 640

            max_scaled_side_length = (min(rendered_height, rendered_width)) * .9
            max_map_side = max(field_map.get_width(), field_map.get_length())

            scale_factor = 1
            while ((max_map_side * scale_factor) > max_scaled_side_length):
                scale_factor -= 0.05

            logging.getLogger(__name__).info(f"Rendered map will be scaled down to {scale_factor}")

            self.__image_scaler = FieldScaler(field_map=field_map, scaled_height=rendered_height, scaled_width=rendered_width, scale_factor=scale_factor, invert_x_axis=True, invert_y_axis=True)
        
        return self.__image_scaler


    # returns targets within sight range of the given position
    def get_visible_targets (self, x, y, sight_range):
        visible_targets = []
        logging.getLogger(__name__).info(f"Checking within {sight_range} dist from {x},{y} for target")
        for t in self.__targets:
            if self.__field_scaler.is_coord_visible (sim_perspective_x = x, sim_perspective_y = y, sim_target_x = t.pos[0], sim_target_y = t.pos[1], sight_range_sim=sight_range):
                logging.getLogger(__name__).info("Target is visible")
                visible_targets.append(t)
            else:
                logging.getLogger(__name__).info(f"Target at {t.pos} is not visible")


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

        for a in self.__search_agents:
            a.notify_event ('found')

    def run_model(self, step_count=200):
        logging.getLogger(__name__).info(f"Search Agents: {self.schedule.get_type_count(SearchAgent)}")

        for i in range(step_count):
            if self.__found:
                return

            self.step()


