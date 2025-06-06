#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of CARS
# (see https://github.com/CNES/cars).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains function for holes detection.
"""

# Third party imports
import numpy as np

from cars.core import constants as cst


def classif_to_stacked_array(disp_map, class_index):
    """
    Convert disparity dataset to mask correspoding to all classes

    :param disp_map: disparity dataset
    :type disp_map: xarray Dataset
    :param class_index: classification tags
    :type class_index: list of str

    """

    index_class = np.where(
        np.isin(
            np.array(disp_map.coords[cst.BAND_CLASSIF].values),
            np.array(class_index),
        )
    )[0].tolist()
    # get index for each band classification of the non zero values
    stack_index = np.any(
        disp_map[cst.EPI_CLASSIFICATION].values[index_class, :, :] > 0,
        axis=0,
    )

    return stack_index
