import traceback
from pathlib import Path
from typing import List

import os
import pdb
import numpy as np
import pandas as pd
import torch
from argodataset.argoverse.map_representation.map_api import ArgoverseMap

from . import interpolate as interp_utils
from .av1_data_utils import(
    OBJECT_TYPE_MAP,
    OBJECT_TYPE_MAP_COMBINED,
    LaneTypeMap,
)

class Av1Extractor:
    def __init__(self,
                 radius: float,
                 save_path: Path = None,
                 mode: str = "train",
                 ignore_type: List[int] = [5, 6, 7, 8, 9],
                 remove_outlier_actors: bool = True,
                ) -> None:
        self.save_path = save_path
        self.mode = mode
        self.radius = radius
        self.remove_outlier_actors = remove_outlier_actors
        self.ignore_type = ignore_type

    def save(self, 
             file: Path,
             am: ArgoverseMap,):
        assert self.save_path is not None

        try:
            data = self.get_data(file, am)
        except Exception:
            print(traceback.format_exc())
            print("found error while extracting data from {}".format(file))
        save_file = self.save_path / (file.stem + ".pt")
        torch.save(data, save_file)


    
        

    
