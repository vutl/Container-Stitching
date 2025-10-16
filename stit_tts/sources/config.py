import json
import os
from os.path import dirname


# pointer to outside of gsan module
ROOT = dirname(dirname(os.path.abspath(__file__)))
with open(f"{ROOT}/resources/config/config.json") as f:
    cfg = json.load(f)

with open(f"{ROOT}/resources/config/models.json") as f:
    models_cfg = json.load(f)

LINE_CONFIG = models_cfg["lineConfig"]
DENT_CONFIG = models_cfg["dentConfig"]
BLACKBOX_CONFIG = models_cfg["blackBoxConfig"]
CORNERS_CONFIG = models_cfg["cornersConfig"]
CORNERS_TOP_CONFIG = models_cfg["cornersTopConfig"]
SUPERGLUE_CONFIG = models_cfg["superGlueConfig"]

SAVE_ROOT = cfg["saveRoot"]
HOST = cfg["host"]
PORT = cfg["port"]
WORKERS = cfg["workers"]

DROP = cfg["drop"]
TIMEOUT_KEEP_ALIVE = cfg["timeoutKeepAlive"]
MAX_PIXEL_DISTANCE = cfg["maxPixelDistance"]
ROOT_STORAGE = cfg["rootStorage"]
DEVICE = cfg["device"]
WORKERS = cfg["workers"]
