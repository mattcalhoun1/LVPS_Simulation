import logging
from lvps.visual.search.server import server

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(module)s:%(message)s', level=logging.INFO)
server.launch(open_browser=True)

