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
Stereo module:
contains stereo-rectification, disparity map estimation
"""

# Standard imports
from __future__ import absolute_import
import warnings

import rasterio as rio
import xarray as xr
from dask import sizeof


# Register sizeof for xarray
# TODO What to we do with this one ?
@sizeof.sizeof.register_lazy("xarray")
def register_xarray():
    """
    Add hook to dask so it correctly estimates memory used by xarray
    """
    @sizeof.sizeof.register(xr.DataArray)
    #pylint: disable=unused-variable
    def sizeof_xarray_dataarray(xarr):
        """
        Inner function for total size of xarray_dataarray
        """
        total_size = sizeof.sizeof(xarr.values)
        for __, carray in xarr.coords.items():
            total_size += sizeof.sizeof(carray.values)
        total_size += sizeof.sizeof(xarr.attrs)
        return total_size
    @sizeof.sizeof.register(xr.Dataset)
    #pylint: disable=unused-variable
    def sizeof_xarray_dataset(xdat):
        """
        Inner function for total size of xarray_dataset
        """
        total_size = 0
        for __, varray in xdat.data_vars.items():
            total_size += sizeof.sizeof(varray.values)
        for __, carray in xdat.coords.items():
            total_size += sizeof.sizeof(carray)
        total_size += sizeof.sizeof(xdat.attrs)
        return total_size

# Filter rasterio warning when image is not georeferenced
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
