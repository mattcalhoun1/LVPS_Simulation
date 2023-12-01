import mesa

from .agents import SearchAgent, Obstacle, Boundary, Target
from .model import AutonomousSearch

color_dic = {4: "#005C00", 3: "#008300", 2: "#00AA00", 1: "#00F800"}


def SearchAgent_portrayal(agent):
    if agent is None:
        return

    if type(agent) is SearchAgent:
        #return {"Shape": "lvps/visual/resources/ant.png", "scale": 0.9, "Layer": 1}
        return {
            "Color": "#11FF11",
            "Shape": "rect",
            "Filled": "true",
            "Layer": 0,
            "w": 2,
            "h": 2,
        }

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
    [{"Label": "TotalDistance", "Color": "#AA0000"}]
)

model_params = {
    # The following line is an example to showcase StaticText.
    "title": mesa.visualization.StaticText("Parameters:"),
    #"grass": mesa.visualization.Checkbox("Grass Enabled", True),
    #"grass_regrowth_time": mesa.visualization.Slider("Grass Regrowth Time", 20, 1, 50),
    #"initial_sheep": mesa.visualization.Slider(
    #    "Initial Sheep Population", 100, 10, 300
    #),
    #"sheep_reproduce": mesa.visualization.Slider(
    #    "Sheep Reproduction Rate", 0.04, 0.01, 1.0, 0.01
    #),
    "num_robots": mesa.visualization.Slider("Number of Robots", 1, 1, 10),
    "num_targets": mesa.visualization.Slider("Number of Targets", 1, 1, 5),
    #"wolf_reproduce": mesa.visualization.Slider(
    #    "Wolf Reproduction Rate",
    #    0.05,
    #    0.01,
    #    1.0,
    #    0.01,
    #    description="The rate at which wolf agents reproduce.",
    #),
    #"wolf_gain_from_food": mesa.visualization.Slider(
    #    "Wolf Gain From Food Rate", 20, 1, 50
    #),
    #"sheep_gain_from_food": mesa.visualization.Slider("Sheep Gain From Food", 4, 1, 10),
}

server = mesa.visualization.ModularServer(
    AutonomousSearch, [canvas_element, chart_element], "LVPS Autonomous Search Simulation", model_params
)
# server.launch()
