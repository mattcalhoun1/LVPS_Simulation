import mesa
import random
import logging

from .agents import SearchAgent, Target, Obstacle, Boundary
from lvps.simulation.lvps_sim_environment import LvpsSimEnvironment
from field.field_scaler import FieldScaler
from field.field_renderer import FieldRenderer
from lvps.simulation.simulated_agent import SimulatedAgent
from lvps.simulation.sim_events import SimEventType
from lvps.simulation.agent_types import AgentTypes
from ...strategies.reasonable_search_strategy import ReasonableSearchStrategy
from ...strategies.rl_search_strategy import RLSearchStrategy
from ...strategies.random_search_strategy import RandomSearchStrategy

import numpy as np

class AutonomousSearch(mesa.Model):
    num_robots = 1
    num_targets = 1

    def __init__(self, width=120, height=120, num_robots=1, num_targets=1):

        # Set parameters
        self.__field_scaler = None
        self.width = width
        self.height = height
        self.num_robots = num_robots
        self.num_targets = num_targets
        self.__search_agents = {}
        self.__lvps_env = None
        self.__next_agent_id = 0
        self.__grayscale = True

        logging.getLogger(__name__).info(f"AutonomousSearch mesa model height: {height}, width: {width}")

        self.__found_targets = []

        self.schedule = mesa.time.RandomActivationByType(self)
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)
        self.datacollector = mesa.DataCollector({
            "TotalDistance": lambda m: m.get_total_distance_traveled()
            },
        )

        self.add_obstacles_and_boundaries()
        self.add_targets()
        self.add_search_agents()

        self.get_lvps_environment().add_event_subscription (event_type = SimEventType.AgentMoved, listener = self)
        self.get_lvps_environment().add_event_subscription (event_type = SimEventType.TargetFound, listener = self)

        self.running = True
        self.datacollector.collect(self)

    def __get_unique_id (self):
        self.__next_agent_id += 1
        return self.__next_agent_id

    def add_obstacles_and_boundaries (self):
        sim_scaler = self.get_field_sim_scaler()
        for _, (x, y) in self.grid.coord_iter():
            # if there's a barrier here, add it
            if not sim_scaler.is_in_bounds(x,y):
                boundary = Boundary(self.__get_unique_id(), (x,y), self)
                self.grid.place_agent(boundary, (x, y))
            elif sim_scaler.is_obstacle(x,y):
                obstacle = Obstacle(self.__get_unique_id(), (x,y), self)
                self.grid.place_agent(obstacle, (x, y))

    def add_search_agents (self):
        # Create search agents, drop onto random spots

        # the agents get a common field renderer, which allows them to export PNG files of the sim, with optional awareness of other agents' location
        field_renderer = FieldRenderer(field_map = self.get_lvps_environment().get_map(), map_scaler=self.get_lvps_environment().get_field_image_scaler(), grayscale=self.__grayscale)
        for i in range(self.num_robots):
            lvps_x,lvps_y = self.get_field_sim_scaler().get_random_traversable_coords()
            lvps_heading = random.randrange(-1800,1800)/10 # pick a random starting heading
            new_agent_id = self.__get_unique_id()

            # this agent receives a paired agent simulation
            lvps_agent = SimulatedAgent(
                agent_id=new_agent_id, 
                agent_type=np.random.choice([AgentTypes.MecCar, AgentTypes.Tank]),
                field_renderer=field_renderer,
                lvps_env=self.get_lvps_environment())

            # add the agent to the lvps simulation
            self.get_lvps_environment().add_agent (lvps_agent, lvps_x, lvps_y, lvps_heading)

            # add the agent to the visual simulation
            x,y = self.get_field_sim_scaler().get_scaled_coords(lvps_x=lvps_x, lvps_y=lvps_y)
            x = round(x)
            y = round(y)

            logging.getLogger(__name__).info(f"Search Agent: LVPS ({lvps_x},{lvps_y}), Scaled ({x},{y})")

            sa = SearchAgent(
                self.__get_unique_id(),
                (x, y),
                self,
                self.get_field_sim_scaler(),
                agent_strategy=self.__get_agent_strategy(),
                lvps_agent=lvps_agent)

            self.grid.place_agent(sa, (x, y))
            self.schedule.add(sa)
            self.__search_agents[new_agent_id] = sa

    def add_targets (self):
        # create targets
        for i in range(self.num_targets):
            lvps_x,lvps_y = self.get_field_sim_scaler().get_random_traversable_coords()
            x,y = self.get_field_sim_scaler().get_scaled_coords(lvps_x=lvps_x, lvps_y=lvps_y)
            x = round(x)
            y = round(y)

            logging.getLogger(__name__).info(f"Traversable: LVPS ({lvps_x},{lvps_y}), Scaled ({x},{y})")

            agent_id = self.__get_unique_id()
            target_agent = Target(agent_id, (x,y), self)
            self.grid.place_agent(target_agent, (x, y))
            self.schedule.add(target_agent)
            self.get_lvps_environment().add_target(
                target_id=agent_id,
                target_name=f'coin_{agent_id}',
                target_type='coin',
                target_x=lvps_x,
                target_y=lvps_y)

    def get_total_distance_traveled (self):
        total = 0
        for a in self.__search_agents:
            agent = self.__search_agents[a]
            total += agent.get_lvps_agent().get_total_distance_traveled()
        return total

    def get_field_sim_scaler (self):
        if self.__field_scaler is None:
            self.__field_scaler = FieldScaler(
                field_map=self.get_lvps_environment().get_map(), 
                scaled_height=self.height - 1, 
                scaled_width=self.width - 1, 
                scale_factor=1
            )
        
        return self.__field_scaler
    
    def get_lvps_environment (self):
        if self.__lvps_env is None:
            self.__lvps_env = LvpsSimEnvironment()
        return self.__lvps_env

    def handle_event (self, event_type, event_details):
        if event_type == SimEventType.AgentMoved:
            # update the view
            agent_id = event_details['agent_id']
            lvps_x = event_details['x']
            lvps_y = event_details['y']
            heading = event_details['heading']

            sim_x, sim_y = self.__field_scaler.get_scaled_coords(lvps_x=lvps_x, lvps_y=lvps_y)
            visual_agent = self.__search_agents[agent_id]
            logging.getLogger(__name__).debug(f"handle_event moving agent {agent_id} to sim position: {round(sim_x)},{round(sim_y)}")
            
            # if it's out of the image, move to the edge
            if sim_x < 0:
                sim_x = 0
            elif sim_x >= self.__field_scaler.get_scaled_width():
                sim_x = self.__field_scaler.get_scaled_width() - 1

            if sim_y < 0:
                sim_y = 0
            elif sim_y >= self.__field_scaler.get_scaled_height():
                sim_y = self.__field_scaler.get_scaled_height() - 1

            new_pos = (round(sim_x), round(sim_y))
            self.grid.move_agent(visual_agent, new_pos)
        elif event_type == SimEventType.TargetFound:
            target_id = event_details['target_id']
            if target_id not in self.__found_targets:
                self.__found_targets.append(target_id)

            if len(self.__found_targets) >= self.num_targets:
                logging.getLogger(__name__).info("All targets found. search is complete")

    def __get_agent_strategy (self):
        #return RandomSearchStrategy()
        #return ReasonableSearchStrategy(render_field=False)
        return RLSearchStrategy(environment=self.__lvps_env, model_file='/home/matt/projects/lvps_rl_models/bw/best_model.zip')

    def step(self):
        if len(self.__found_targets) < self.num_targets:
            self.schedule.step()
            self.datacollector.collect(self)

    def run_model(self, step_count=200):
        logging.getLogger(__name__).info(f"Search Agents: {self.schedule.get_type_count(SearchAgent)}")

        for i in range(step_count):
            if len(self.__found_targets) >= self.num_targets:
                return

            self.step()


