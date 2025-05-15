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

import copy

# Standard imports
import logging
import os
import subprocess
import sys

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def set_dask_config():
    """
    Set dask config path
    """

    # Get cluster file path out of current python process
    cmd = [
        "python",
        "-c",
        "from  cars.orchestrator import cluster; "
        "import os; print(os.path.dirname(cluster.__file__))",
    ]
    try:
        cmd_output = subprocess.run(cmd, capture_output=True, check=True).stdout
        cluster_path = str(cmd_output)[2:-3]
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"{err} {err.stderr.decode('utf8')}") from err
    # Force the use of CARS dask configuration
    dask_config_path = os.path.join(
        cluster_path,
        "dask_config",
    )

    if not os.path.isdir(dask_config_path):
        raise NotADirectoryError(
            "Wrong dask config path: {}".format(dask_config_path)
        )
    os.environ["DASK_CONFIG"] = str(dask_config_path)


# Set dask config before cars imports
set_dask_config()

# fmt: off
# isort: off
# pylint: disable=C0413, E0401
from cars.applications.grid_generation import (  # noqa: E402
    grid_correction,
)
from cars.data_structures import (  # noqa: E402
    corresponding_tiles_tools,
)
from cars.data_structures.cars_dataset import (  # noqa: E402
    load_dict,
)
# pylint: enable=C0413

# fmt: on
# isort: on


def compute_cell(orchestrator, list_cars_ds):
    """
    Compute notebook cell if orchestrator is not sequential.
    Replace Delayed with clear data

    :param orchestrator: orchestrator used
    :param list_cars_ds: list of CarsDataset to compute

    """

    # add cars datasets to save lists
    for cars_ds in list_cars_ds:
        orchestrator.add_to_replace_lists(cars_ds)

    # trigger computation and replacement
    orchestrator.breakpoint()


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


def get_full_data(cars_ds, tag, nodata=-32768):
    """
    Get combined data of CarsDataset

    :param cars_ds: cars dataset to use
    :param tag: key to get from xr.Datasets

    :return: array of full data

    """

    if cars_ds.dataset_type != "arrays":
        logging.error("Not an arrays CarsDataset")
        raise RuntimeError("Not an arrays CarsDataset")

    list_tiles = []
    window = None
    overlap = None

    for row in range(cars_ds.shape[0]):
        for col in range(cars_ds.shape[1]):
            if cars_ds[row, col] is not None:
                if window is None:
                    # first non None tile found
                    window = cars_ds.tiling_grid[row, col, :]
                    overlap = cars_ds.overlaps[row, col, :]
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

    if merged_dataset[0] is None:
        return None

    if tag not in merged_dataset[0]:
        raise RuntimeError("Tag {} not in dataset".format(tag))
    array = merged_dataset[0][tag].values

    if len(array.shape) == 3:
        array = np.rollaxis(array, 0, 3)

    if np.issubdtype(array.dtype, np.floating):
        array[array == nodata] = np.nan

    return array


def cars_cmap():
    """
    Define CARS color maps

    """

    colors = ["navy", "lightsteelblue", "snow", "lightcoral", "red"]
    nodes = [0.0, 0.4, 0.45, 0.5, 1.0]
    cmap_shift = LinearSegmentedColormap.from_list(
        "mycmap", list(zip(nodes, colors))  # noqa: B905
    )

    return cmap_shift


def pandora_cmap():
    """
    Define pandora color map

    """

    colors = ["crimson", "lightpink", "white", "yellowgreen"]
    nodes = [0.0, 0.4, 0.5, 1.0]
    cmap_shift = LinearSegmentedColormap.from_list(
        "mycmap", list(zip(nodes, colors))  # noqa: B905
    )

    return cmap_shift


def configure_matplotib_notebook():
    """
    Configure matplotlib
    """
    if "ipykernel" in sys.modules:
        try:
            from IPython import get_ipython  # pylint: disable=C0415

            get_ipython().run_line_magic("matplotlib", "inline")
        except Exception:
            pass


def show_data(data, figsize=(11, 11), mode=None):
    """
    Show data with matplotlib


    available mode : "dsm", "image",
    """
    configure_matplotib_notebook()
    # squeeze data
    data = copy.deepcopy(data)
    data = np.squeeze(data)

    # Replace Nan by 0 for visualisation
    data[np.isnan(data)] = 0

    if mode in ("dsm", "image"):
        data[data < 0] = 0

    if np.min(data) < 0 or np.max(data) > 1:
        # crop

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
    desc = None

    # Save tiles
    for row in range(cars_ds.shape[0]):
        for col in range(cars_ds.shape[1]):
            if cars_ds[row, col] is not None:
                if desc is None:
                    desc = cars_ds.generate_descriptor(
                        cars_ds[row, col],
                        file_name,
                        tag=tag,
                        dtype=dtype,
                        nodata=nodata,
                    )
                cars_ds.run_save(
                    cars_ds[row, col],
                    file_name,
                    tag=tag,
                    descriptor=desc,
                )

    # close descriptor
    desc.close()


def show_epipolar_images(
    img_left, mask_left, img_right, mask_right, fig_size=8
):
    """
    Show both epipolar image side by side

    """
    clip_percent = 5
    vmin_left = np.percentile(img_left, clip_percent)
    vmax_left = np.percentile(img_left, 100 - clip_percent)
    vmin_right = np.percentile(img_right, clip_percent)
    vmax_right = np.percentile(img_right, 100 - clip_percent)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(fig_size, 1.05 * fig_size / 2),
        subplot_kw={"aspect": 1},
    )
    axes[0].set_title("Left image")
    axes[0].imshow(
        img_left,
        cmap="gray",
        interpolation="spline36",
        vmin=vmin_left,
        vmax=vmax_left,
    )
    axes[0].imshow(
        np.ma.masked_where(mask_left == 0, mask_left), cmap="tab10", alpha=0.5
    )
    axes[0].axhline(len(img_left) / 2.0, color="red")
    axes[1].set_title("Right image")
    axes[1].imshow(
        img_right,
        cmap="gray",
        interpolation="spline36",
        vmin=vmin_right,
        vmax=vmax_right,
    )
    axes[1].imshow(
        np.ma.masked_where(mask_right == 0, mask_right), cmap="tab10", alpha=0.5
    )
    axes[1].axhline(len(img_right) / 2.0, color="red")
    fig.tight_layout()


def update_advanced_conf_with_a_priori(
    advanced_conf, content_json_with_a_priori, input_dir_path
):
    """
    Update given advanced parameter dict with a priori in .json file

    :param advanced_conf: dict to overide
    :type advanced_conf: dict
    :param content_json_with_a_priori: json file to get a priori from
    :type content_json_with_a_priori: str

    """
    # Get a priori in file
    a_priori_dict_full = load_dict(content_json_with_a_priori)

    epipolar_a_priori = a_priori_dict_full["advanced"]["epipolar_a_priori"]
    terrain_a_priori = a_priori_dict_full["advanced"]["terrain_a_priori"]

    # Overide paths
    for key in ["dem_median", "dem_min", "dem_max"]:
        if not os.path.isfile(terrain_a_priori[key]):
            terrain_a_priori[key] = os.path.join(
                input_dir_path,
                terrain_a_priori[key],
            )

    # set in conf
    advanced_conf["epipolar_a_priori"] = epipolar_a_priori
    advanced_conf["terrain_a_priori"] = terrain_a_priori
    advanced_conf["use_epipolar_a_priori"] = True


def extract_a_priori_from_config(conf):
    """
    Extract a priori from configuration
    """

    if "epipolar_a_priori" not in conf:
        raise RuntimeError("Epipolar a priori not set")

    epipolar_a_priori = conf["epipolar_a_priori"]
    key = list(epipolar_a_priori.keys())[0]

    grid_coefficients = epipolar_a_priori[key]["grid_correction"]
    disparity_range = epipolar_a_priori[key]["disparity_range"]

    terrain_a_priori = conf["terrain_a_priori"]
    dem_median = terrain_a_priori["dem_median"]
    dem_min = terrain_a_priori["dem_min"]
    dem_max = terrain_a_priori["dem_max"]

    return grid_coefficients, disparity_range, dem_median, dem_min, dem_max


def apply_grid_correction(grid, grid_coefficients, save_folder):
    """
    Correct grid with grid correction

    """
    if grid_coefficients in (None, []):
        raise RuntimeError("No grid correction provided")

    # Correct grid right with provided epipolar a priori
    return grid_correction.correct_grid_from_1d(
        grid, list(grid_coefficients), False, save_folder
    )
