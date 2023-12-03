import logging
import math
import numpy as np
import random
from lvps.generators.static_field_map_generator import StaticFieldMapGenerator
from lvps.generators.random_field_map_generator import RandomFieldMapGenerator
from field.field_scaler import FieldScaler
from trig.trig import BasicTrigCalc
from position.confidence import Confidence
from lvps.strategies.agent_actions import AgentActions
from lvps.simulation.sim_events import SimEventSubscriptions, SimEventType

class LvpsSimEnvironment:
    def __init__(self):
        self.__targets = {}
        self.__found_targets = {}
        self.__agents = {}

        self.__target_find_position_threshold = 0.07 # 'found' position has to be within this distance in order to be considered found
        self.__agent_collision_threshold = 0.1 # can't be this percent close to any other agent
        self.__map = None
        self.__image_scaler = None

        self.__event_subscriptions = SimEventSubscriptions()
        self.__trig_calc = BasicTrigCalc()

    def add_event_subscription (self, event_type, listener):
        self.__event_subscriptions.add_subscription(event_type, listener)

    def get_agent_position (self, agent_id):
        agent = self.__agents[agent_id]
        return agent['x'], agent['y'], agent['heading']

    # this simulates the get_coords_and_heading method of pilot.
    # this method is in the environment sim reather than agent sim, because agent should never have
    # access to the exact position
    def estimate_agent_position(self, agent_id):
        est_x = None
        est_y = None
        est_heading = None
        confidence = None

        agent = self.__agents[agent_id]

        # if agent is in a dead spot, positioning will always fail
        is_dead_spot, dead_spot_id = self.__map.is_in_dead_spot (agent['x'], agent['y'], agent['heading'])
        if is_dead_spot:
            logging.getLogger(__name__).info(f"Agent {agent_id} is dead spot {dead_spot_id}, positioning will fail")
        elif self.__does_event_happen(AgentActions.SuccessRate[AgentActions.EstimatePosition]):
            # returns an estimate of the position, with variability

            confidence = np.random.choice([Confidence.CONFIDENCE_HIGH, Confidence.CONFIDENCE_MEDIUM])
            est_x = self.__get_less_accurate(agent['x'], AgentActions.Accuracy[AgentActions.EstimatePosition][confidence], range_val=self.get_map().get_width())
            est_y = self.__get_less_accurate(agent['y'], AgentActions.Accuracy[AgentActions.EstimatePosition][confidence], range_val=self.get_map().get_length())
            est_heading = self.__get_less_accurate(agent['heading'], AgentActions.Accuracy[AgentActions.Heading][confidence], range_val=360)

            logging.getLogger(__name__).debug(f"Agent {agent_id} positiong successful. ({est_x},{est_y}) : {round(est_heading,1)} deg ")
        else:
            logging.getLogger(__name__).debug(f"Positioning failed for agent {agent_id}")

        return est_x, est_y, est_heading, confidence

    def rotate (self, agent_id, degrees):
        if self.__does_event_happen(AgentActions.SuccessRate[AgentActions.Rotate]):
            actual_adjust = self.__get_less_accurate(degrees, AgentActions.Accuracy[AgentActions.Rotate], range_val=360)
            curr_heading = self.__agents[agent_id]['heading']
            new_heading = curr_heading + actual_adjust
            if new_heading > 180:
                new_heading = -1 * (360 - new_heading)
            elif new_heading < -180:
                new_heading = 360 - abs(new_heading)
            self.__agents[agent_id]['heading'] = new_heading

            self.__event_subscriptions.notify_subscribers(SimEventType.AgentRotated, {'agent_id':agent_id, 'heading':new_heading})

            logging.getLogger(__name__).debug(f"{agent_id} wanted go rotate {degrees}, actual: {actual_adjust}, old: {curr_heading}, new: {new_heading}")
        return False

    # moves forward or backward without affecting rotation
    def __go_straight (self, agent_id, forward, distance, success_rate, accuracy, heading_offset = 0):
        agent = self.__agents[agent_id]
        starting_x = agent['x']
        starting_y = agent['y']
        heading = agent['heading']

        adjusted_heading = heading + heading_offset
        if adjusted_heading > 180:
            adjusted_heading = 180 - abs(180 - adjusted_heading)
        elif adjusted_heading < -180:
            adjusted_heading = 180 - abs(180 + adjusted_heading)

        if self.__does_event_happen(success_rate):
            # adjust the distance a little. The adjustment range may need to be higher
            adjusted_distance = self.__get_less_accurate(distance, accuracy=accuracy, range_val=(distance * 1.5))

            # find the point in the desired direction
            next_x, next_y = self.__get_next_position(heading=adjusted_heading, x=starting_x, y=starting_y, distance=adjusted_distance, forward=forward)

            logging.getLogger(__name__).debug(f"Facing {round(heading)} - going {'forward' if forward else 'reverse'} by {round(distance)} would end up at ({round(next_x)}, {round(next_y)})")

            # if path is not open, make the new position on the obstacle, so the agent is penalized
            is_blocked, obstacle_id = self.__map.is_path_blocked(starting_x, starting_y, next_x, next_y, agent['agent'].get_path_width())
            if is_blocked:
                # put in the center of the obstacle
                logging.getLogger(__name__).info(f"Agent {agent_id} ran into obstacle {obstacle_id}")
                o_xmin, o_ymin, o_xmax, o_ymax = self.__map.get_obstacle_bounds(obstacle_id)
                self.__agents[agent_id]['x'] = o_xmin + .5 * (o_xmax - o_xmin)
                self.__agents[agent_id]['y'] = o_ymin + .5 * (o_ymax - o_ymin)
                self.__agents[agent_id]['heading'] = heading
            else:
                logging.getLogger(__name__).debug(f"Agent {agent_id} successfully traveled from {self.__agents[agent_id]['x']},{self.__agents[agent_id]['y']} to {next_x,next_y} and is still facing {heading}")
                self.__agents[agent_id]['x'] = next_x
                self.__agents[agent_id]['y'] = next_y
                self.__agents[agent_id]['heading'] = heading

            self.__event_subscriptions.notify_subscribers(SimEventType.AgentMoved, {'agent_id':agent_id, 'x':next_x, 'y':next_y, 'heading':heading})
            return True
        return False
    
    def go_forward (self, agent_id, distance):
        logging.getLogger(__name__).debug(f"Agent {agent_id} going forward for distance {distance}")
        return self.__go_straight(
            agent_id=agent_id,
            forward=True,
            distance=distance,
            success_rate=AgentActions.SuccessRate[AgentActions.GoForward],
            accuracy=AgentActions.SuccessRate[AgentActions.GoForward])

    def go_reverse (self, agent_id, distance):
        logging.getLogger(__name__).debug(f"Agent {agent_id} going reverse for distance {distance}")
        return self.__go_straight(
            agent_id=agent_id,
            forward=False,
            distance=distance,
            success_rate=AgentActions.SuccessRate[AgentActions.GoReverse],
            accuracy=AgentActions.SuccessRate[AgentActions.GoReverse])

    def strafe_left (self, agent_id, distance):
        logging.getLogger(__name__).debug(f"Agent {agent_id} strafing left for distance {distance}")
        return self.__go_straight(
            agent_id=agent_id,
            forward=True,
            distance=distance,
            success_rate=AgentActions.SuccessRate[AgentActions.Strafe],
            accuracy=AgentActions.SuccessRate[AgentActions.Strafe],
            heading_offset=-90.0)

    def strafe_right (self, agent_id, distance):
        logging.getLogger(__name__).debug(f"Agent {agent_id} strafing right for distance {distance}")
        return self.__go_straight(
            agent_id=agent_id,
            forward=True,
            distance=distance,
            success_rate=AgentActions.SuccessRate[AgentActions.Strafe],
            accuracy=AgentActions.SuccessRate[AgentActions.Strafe],
            heading_offset=90.0)

    def __get_next_position (self, heading, x, y, distance, forward = True):
        return self.__trig_calc.get_coords_for_zeronorth_angle_and_distance (
                heading=heading,
                x = x,
                y = y,
                distance = distance,
                is_forward = forward)

    def go (self, agent_id, target_x, target_y):
        logging.getLogger(__name__).debug(f"Agent {agent_id} going toward {target_x},{target_y}")

        adjusted_x = self.__get_less_accurate(target_x, AgentActions.Accuracy[AgentActions.Go], range_val=self.__map.get_width())
        adjusted_y = self.__get_less_accurate(target_y, AgentActions.Accuracy[AgentActions.Go], range_val=self.__map.get_length())

        new_heading = self.__get_relative_heading_end (
            start_x = self.__agents[agent_id]['x'],
            start_y = self.__agents[agent_id]['y'],
            end_x = adjusted_x,
            end_y = adjusted_y)

        if self.__does_event_happen(AgentActions.SuccessRate[AgentActions.Go]):
            logging.getLogger(__name__).debug(f"Agent {agent_id} successfully traveled from {self.__agents[agent_id]['x']},{self.__agents[agent_id]['y']} to {adjusted_x,adjusted_y} and is now facing {new_heading}")
            self.__agents[agent_id]['x'] = adjusted_x
            self.__agents[agent_id]['y'] = adjusted_y
            self.__agents[agent_id]['heading'] = new_heading

            self.__event_subscriptions.notify_subscribers(SimEventType.AgentMoved, {'agent_id':agent_id, 'x':adjusted_x, 'y':adjusted_y, 'heading':new_heading})
            return True
        else:
            logging.getLogger(__name__).debug(f"Agent {agent_id} travel attempt failed")
            return False

    def __get_relative_heading_end (self, start_x, start_y, end_x, end_y):
        slope = 0
        try:
            slope = (end_y - start_y) / (end_x - start_x)
        except:
            # slope is zero  , if divide by zero
            pass
        slope_degrees = math.degrees(math.atan(slope))
        
        # convert to our degree system
        end_heading = 90 - slope_degrees if slope_degrees > 0 else 90 + abs(slope_degrees)

        # If we went to the left, the heading will be the opposite direction of the slope
        if end_x < start_x:
            end_heading = -1 * (180 - end_heading)

        return end_heading

    def is_too_close_to_other_agents (self, agent_id):
        # calculate distance from each other agent
        closest_id, closest_dist = self.__find_closest_agent (agent_id)
        if closest_dist <= min(self.get_map().get_width(), self.get_map().get_length()) * self.__agent_collision_threshold:
            logging.getLogger(__name__).info(f"Agent {agent_id} may have collided with {closest_id}!")
            return True
        return False


    def __get_less_accurate(self, good_value, accuracy, range_val):
        # choose a random amount to mess with the calculation
        tolerance = 1 - accuracy
        adjust_amount = random.randrange(int(-1 * 1000 * tolerance), int(1000 * tolerance)) / 1000
        return good_value + (range_val * adjust_amount)

    # returns true if a "random" event should occur, based on the given occurance rate
    def __does_event_happen (self, occurance_rate):
        return random.randrange(0,100) <= occurance_rate * 100

    def get_field_image_scaler (self):
        if self.__image_scaler is None:
            rendered_height = 320
            rendered_width = 320
            field_map = self.get_map()

            max_scaled_side_length = (min(rendered_height, rendered_width)) * .9
            max_map_side = max(field_map.get_width(), field_map.get_length())

            scale_factor = 1
            while ((max_map_side * scale_factor) > max_scaled_side_length):
                scale_factor -= 0.05

            logging.getLogger(__name__).info(f"Rendered map will be scaled down to {scale_factor}")

            self.__image_scaler = FieldScaler(field_map=field_map, scaled_height=rendered_height, scaled_width=rendered_width, scale_factor=scale_factor, invert_x_axis=True, invert_y_axis=True)
        
        return self.__image_scaler

    def add_agent (self, agent, x, y, heading):
        self.__agents[agent.get_id()] = {
            'agent':agent,
            'x':x,
            'y':y,
            'heading':heading
        }
        logging.getLogger(__name__).info(f"Agent {agent.get_id()} added at ({x},{y}) facing {heading}")

    def get_agent_position(self, agent_id):
        agent = self.__agents[agent_id]
        return agent['x'], agent['y'], agent['heading'], Confidence.CONFIDENCE_HIGH

    def add_target (self, target_id, target_name, target_type, target_x, target_y):
        self.__targets[target_id] = {
            'id':target_id,
            'name':target_name,
            'type':target_type,
            'x':target_x,
            'y':target_y
        }
        logging.getLogger(__name__).info(f"Target {target_id} added at ({target_x},{target_y})")

    # returns targets within sight range of the given agent
    def get_visible_targets (self, agent_id, sight_distance):
        visible_targets = []
        visible_headings = []
        agent = self.__agents[agent_id]['agent']
        agent_x = self.__agents[agent_id]['x']
        agent_y = self.__agents[agent_id]['y']

        logging.getLogger(__name__).debug(f"For agent {agent_id}, checking within {sight_distance} dist from {agent_x},{agent_y} for target")
        for tid in self.__targets:
            target = self.__targets[tid]
            if agent.get_field_renderer().get_map_scaler().is_lvps_coord_visible (
                lvps_perspective_x = agent_x, 
                lvps_perspective_y = agent_y,
                lvps_target_x = target['x'],
                lvps_target_y = target['y'],
                lvps_sight_range=sight_distance):

                # the slope has to NOT be in the agent's blind spot
                relative_degrees = self.__get_relative_heading_end(
                    start_x=agent_x,
                    start_y=agent_y,
                    end_x=target['x'],
                    end_y=target['y']
                )

                if relative_degrees >= agent.get_relative_search_begin() and relative_degrees <= agent.get_relative_search_end():
                    logging.getLogger(__name__).debug(f"Target {target['name']}) is visible to {agent_id}")
                    visible_targets.append(target)
                    visible_headings.append(relative_degrees)
                else:
                    logging.getLogger(__name__).debug(f"Target {target['name']} is close to agent {agent_id}, but in its blind spot")


        return visible_targets, visible_headings

    def get_map (self):
        if self.__map is None:
            #self.__map = StaticFieldMapGenerator().generate_map()
            self.__map = RandomFieldMapGenerator(
                min_width=150,
                max_width=500,
                min_height=150,
                max_height=500,
                min_obstacles=3,
                max_obstacles=10,
                min_obstacle_size_pct=0.01,
                max_obstacle_size_pct=0.25,
                min_deadspots=2,
                max_deadspots=10,
                min_deadspot_size_pct = 0.01,
                max_deadspot_size_pct=0.05
            ).generate_map()

            logging.getLogger(__name__).info(f"Random LVPS Map Height: {self.__map.get_length()}, Width: {self.__map.get_width()}, Boundaries: {self.__map.get_boundaries()}")
        return self.__map

    # tells whether the given target is already found
    def is_target_found (self, agent_id, x, y):
        #logging.getLogger(__name__).info(f"Agent {agent_id} checking if target at ({x},{y}) is already found")
        closest_target = self.__find_closest_target(x, y)

        return closest_target is not None and closest_target in self.__found_targets

    def report_target_found (self, agent_id, x, y):
        logging.getLogger(__name__).info(f"Agent {agent_id} reports finding the target at ({x},{y})")

        closest_target = self.__find_closest_target(x, y)
        if closest_target is not None and closest_target not in self.__found_targets:
            self.__found_targets[closest_target] = self.__targets[closest_target]
            self.__agents[agent_id]['agent'].get_field_renderer().update_search_state (agent_id, self.__targets[closest_target]['type'], x, y)
            self.__event_subscriptions.notify_subscribers(SimEventType.TargetFound, {'agent_id':agent_id, 'target_id':closest_target})

        elif closest_target is None:
            logging.getLogger(__name__).info("There is no target at that location")
        else:
            logging.getLogger(__name__).info(f"Target {closest_target} was successfuly found by {agent_id}")
            return True # inform agent the target found was good
        
        return False

    def get_num_found_targets (self):
        return len(self.__found_targets)

    def __find_closest_target (self, x, y):
        # find the target nearest to the specified coords
        closest_target = None
        closest_dist = None
        for tid in self.__targets:
            this_target = self.__targets[tid]
            this_dist = self.__get_distance(x, y, this_target['x'], this_target['y'])

            if closest_dist is None or closest_dist > this_dist:
                closest_dist = this_dist
                closest_target = tid
        
        if closest_dist is None:
            return None
        elif closest_dist <= (self.__target_find_position_threshold) * min(self.get_map().get_width(), self.get_map().get_length()):
            return closest_target
        else:
            logging.getLogger(__name__).debug(f"Target {closest_target} at a distance of {closest_dist} is higher than the threshold dist:  {(self.__target_find_position_threshold) * min(self.get_map().get_width(), self.get_map().get_length())}")
        return None

    # finds the agent closest to the given agent
    def __find_closest_agent (self, agent_id):
        # find the target nearest to the specified coords
        closest_agent = None
        closest_dist = None
        x = self.__agents[agent_id]['x']
        y = self.__agents[agent_id]['y']

        for aid in self.__agents:
            if aid != agent_id:
                curr_agent = self.__agents[aid]
                this_dist = self.__get_distance(x, y, curr_agent['x'], curr_agent['y'])

                if closest_dist is None or closest_dist > this_dist:
                    closest_dist = this_dist
                    closest_agent = aid
        
        return closest_agent, closest_dist
    

    def __get_distance(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx**2 + dy**2)    

