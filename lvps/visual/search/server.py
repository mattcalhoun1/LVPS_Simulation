import mesa

from .agents import SearchAgent, Obstacle, Boundary, Target
from .model import AutonomousSearch

color_dic = {4: "#005C00", 3: "#008300", 2: "#00AA00", 1: "#00F800"}


def SearchAgent_portrayal(agent):
    if agent is None:
        return

    if type(agent) is SearchAgent:
        return {"Shape": "lvps/visual/resources/ant.png", "scale": 0.9, "Layer": 1}

    elif type(agent) is Boundary:
        return {
            "Color": "#F1F1F1",
            "Shape": "rect",
            "Filled": "true",
            "Layer": 0,
            "w": 1,
            "h": 1,
        }
    
    elif type(agent) is Obstacle:
        return {
            "Color": "#1111EE",
            "Shape": "rect",
            "Filled": "true",
            "Layer": 0,
            "w": 1,
            "h": 1,
        }

    elif type(agent) is Target:
        return {
            "Color": "#FF1111",
            "Shape": "rect",
            "Filled": "true",
            "Layer": 0,
            "w": 1,
            "h": 1,
        }

    return {}


canvas_element = mesa.visualization.CanvasGrid(SearchAgent_portrayal, 120, 120, 500, 500)
chart_element = mesa.visualization.ChartModule(
    [{"Label": "SearchAgent", "Color": "#AA0000"}]
)

server = mesa.visualization.ModularServer(
    AutonomousSearch, [canvas_element, chart_element], "LVPS Autonomous Search Simulation"
)
# server.launch()
