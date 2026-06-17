from pathlib import Path

import pandas as pd

OBJECT_TYPE_MAP = {
    "AV": 0,
    "AGENT": 1,
    "OTHERS": 2,
}

OBJECT_TYPE_MAP_COMBINED = {
    "AV": 0,
    "AGENT": 0,
    "OTHERS": 1,
}

LaneTypeMap = {
    0: 0,
    1: 1,
    2: 2,
}