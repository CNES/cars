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
#
"""
CARS Datasets module
"""

# Standard imports
from typing import List

# Third party imports
import numpy as np
import xarray as xr

# CARS imports
from cars.core import constants as cst

# TODO: refacto constants: define constants here as only concerning datasets

def create_im_dataset(img: np.ndarray,
                      region: List[int],
                      largest_size: List[int],
                      band_coords: bool=False,
                      msk: np.ndarray=None) -> xr.Dataset:
    """
    Create image dataset as used in cars.

    :param img: image as a numpy array
    :param region: region as list [xmin ymin xmax ymax]
    :param largest_size: whole image size
    :param band_coords: set to true to add the coords 'band' to the dataset
    :param msk: image mask as a numpy array (default None)
    :return: The image dataset as used in cars
    """
    nb_bands = img.shape[-1]

    # Add band dimension if needed
    if band_coords or nb_bands > 1:
        bands = range(nb_bands)
        # Reorder dimensions in color dataset in order that the first dimension
        # is band.
        dataset = xr.Dataset({cst.EPI_IMAGE: ([cst.BAND, cst.ROW, cst.COL],
                                     np.einsum('ijk->kij', img)
                                     )},
                             coords={cst.BAND: bands,
                                     cst.ROW: np.array(range(region[1],
                                                             region[3])),
                                     cst.COL: np.array(range(region[0],
                                                             region[2]))
                                     })
    else:
        dataset = xr.Dataset({cst.EPI_IMAGE: ([cst.ROW, cst.COL],
                             img[:, :, 0])},
                             coords={cst.ROW: np.array(range(region[1],
                                                             region[3])),
                                     cst.COL: np.array(range(region[0],
                                                             region[2]))})

    if msk is not None:
        dataset[cst.EPI_MSK] = xr.DataArray(msk.astype(np.int16),
                                            dims=[cst.ROW,
                                            cst.COL])

    dataset.attrs[cst.EPI_FULL_SIZE] = largest_size
    dataset.attrs['region'] = np.array(region)

    return dataset
