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
this module contains functions helpers used in notebooks.
"""

# Standard imports
import logging
import os

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from cars.data_structures import (  # pylint: disable=E0401
    corresponding_tiles_tools,
)


def get_dir_path():
    """
    Get the path of current directory
    """
    return os.path.dirname(__file__)


def set_up_demo_inputs(demo_data_name="data_gizeh_small"):
    """
    Set up demo input, return path

    :param demo_data_name: use demo data name for .tar.bz2 and path to return
    :return: full absolute path of demo data
    """

    inputs_dir = os.path.dirname(__file__)
    inputs_tar = os.path.join(inputs_dir, "{}.tar.bz2".format(demo_data_name))

    # decompact samples demo data
    cmd = "tar xfj {} -C {}".format(inputs_tar, inputs_dir)
    os.system(cmd)

    return os.path.join(inputs_dir, demo_data_name)


def mkdir(root_dir, name_dir):
    """
    Make a directory in root directory
    Returns the full path directory created
    """
    try:
        full_path_name_dir = os.path.join(root_dir, name_dir)
        os.mkdir(full_path_name_dir)
    except OSError:
        logging.warning("Error mkdir {} ".format(full_path_name_dir))
    return full_path_name_dir


def get_full_data(cars_ds, tag):
    """
    Get combined data of CarsDataset

    :param cars_ds: cars dataset to use
    :param tag: key to get from xr.Datasets

    :return: array of full data

    """

    if cars_ds.dataset_type != "arrays":
        logging.error("Not an arrays CarsDataset")
        raise Exception("Not an arrays CarsDataset")

    list_tiles = []
    window = cars_ds.tiling_grid[0, 0, :]
    overlap = cars_ds.overlaps[0, 0, :]

    for row in range(cars_ds.shape[0]):
        for col in range(cars_ds.shape[1]):
            list_tiles.append(
                (
                    cars_ds.tiling_grid[row, col, :],
                    cars_ds.overlaps[row, col, :],
                    cars_ds[row, col],
                )
            )

    merged_dataset = corresponding_tiles_tools.reconstruct_data(
        list_tiles, window, overlap
    )

    array = merged_dataset[0][tag].values

    if len(array.shape) == 3:
        array = np.rollaxis(array, 0, 3)

    return array


def cars_cmap():
    """
    Define CARS color maps

    """

    colors = ["navy", "lightsteelblue", "snow", "lightcoral", "red"]
    nodes = [0.0, 0.4, 0.45, 0.5, 1.0]
    cmap_shift = LinearSegmentedColormap.from_list(
        "mycmap", list(zip(nodes, colors))
    )

    return cmap_shift


def pandora_cmap():
    """
    Define pandora color map

    """

    colors = ["crimson", "lightpink", "white", "yellowgreen"]
    nodes = [0.0, 0.4, 0.5, 1.0]
    cmap_shift = LinearSegmentedColormap.from_list(
        "mycmap", list(zip(nodes, colors))
    )

    return cmap_shift


def show_data(data, figsize=(11, 11), mode=None):
    """
    Show data with matplotlib


    available mode : "dsm", "image"
    """

    if mode in ("dsm", "image"):
        data[data < 0] = 0

    p1 = np.percentile(data, 5)
    p2 = np.percentile(data, 95)

    data[data < p1] = p1
    data[data > p2] = p2

    plt.figure(figsize=figsize)

    cmap = None
    if len(data.shape) == 2:
        cmap = pandora_cmap()

        if mode in ("dsm", "image"):
            cmap = cmap = "gray"

    else:
        data = data / np.max(data)

    imgplot = plt.imshow(data)
    if cmap is not None:
        imgplot.set_cmap(cmap)

    plt.show()


def save_data(cars_ds, file_name, tag, dtype="float32", nodata=-9999):
    """
    Save CarsDataset

    """

    # create descriptor
    desc = cars_ds.generate_descriptor(
        cars_ds[0, 0],
        file_name,
        tag=tag,
        dtype=dtype,
        nodata=nodata,
    )

    # Save tiles
    for row in range(cars_ds.shape[0]):
        for col in range(cars_ds.shape[1]):
            cars_ds.run_save(
                cars_ds[row, col],
                file_name,
                tag=tag,
                descriptor=desc,
            )

    # close descriptor
    desc.close()
