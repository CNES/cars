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
    :parap tag: key to get from xr.Datasets

    :return: array of full data

    """

    if cars_ds.dataset_type != "arrays":
        logging.error("Not an arrays CarsDataset")
        raise Exception("Not an arrays CarsDataset")

    # Get number of bands
    nb_bands = 0
    for row in range(cars_ds.shape[0]):
        for col in range(cars_ds.shape[1]):
            if cars_ds[row, col] is not None:
                if tag not in cars_ds[row, col]:
                    raise Exception("tag not in dataset")
                if len(cars_ds[row, col][tag].values.shape) == 2:
                    nb_bands = 1
                else:
                    nb_bands = cars_ds[row, col][tag].values.shape[0]

                break

    # Create array
    nb_rows = int(np.max(cars_ds.tiling_grid[:, :, 1]))
    nb_cols = int(np.max(cars_ds.tiling_grid[:, :, 3]))

    if nb_bands == 1:
        array = np.empty((nb_rows, nb_cols))
    else:
        array = np.empty((nb_rows, nb_cols, nb_bands))

    # fill array
    windows = cars_ds.tiling_grid.astype(int)
    overlaps = cars_ds.overlaps.astype(int)
    for row in range(cars_ds.shape[0]):
        for col in range(cars_ds.shape[1]):
            row_start = windows[row, col, 0]
            row_end = windows[row, col, 1]
            col_start = windows[row, col, 2]
            col_end = windows[row, col, 3]

            data_with_overlaps = cars_ds[row, col][tag].values
            data_with_overlaps = np.squeeze(data_with_overlaps)
            if len(data_with_overlaps.shape) == 2:
                nb_rows, nb_cols = (
                    data_with_overlaps.shape[0],
                    data_with_overlaps.shape[1],
                )
                current_data = data_with_overlaps[
                    overlaps[row, col, 0] : nb_rows - overlaps[row, col, 1],
                    overlaps[row, col, 2] : nb_cols - overlaps[row, col, 3],
                ]
                array[row_start:row_end, col_start:col_end] = current_data
            else:
                nb_rows, nb_cols = (
                    data_with_overlaps.shape[1],
                    data_with_overlaps.shape[2],
                )
                current_data = data_with_overlaps[
                    :,
                    overlaps[row, col, 0] : nb_rows - overlaps[row, col, 1],
                    overlaps[row, col, 2] : nb_cols - overlaps[row, col, 3],
                ]

                array[row_start:row_end, col_start:col_end, :] = np.rollaxis(
                    current_data, 0, 3
                )

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
