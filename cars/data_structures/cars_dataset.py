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
cars_dataset module:

"""
# pylint: disable=too-many-lines

import copy
import json
import logging
import math

# Standard imports
import os
import pickle
from typing import Dict

# Third party imports
import numpy as np
import pandas
import rasterio as rio
import xarray as xr
from rasterio.profiles import DefaultGTiffProfile
from rasterio.windows import Window

# CARS imports
from cars.core import constants as cst
from cars.core import outputs
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dict, dataframe_converter

# cars dataset dtype
CARS_DS_TYPE_ARRAY = "arrays"
CARS_DS_TYPE_POINTS = "points"
CARS_DS_TYPE_DICT = "dict"

# cars_dataset names
TILES_INFO_FILE = "tiles_info.json"
OVERLAP_FILE = "overlaps.npy"
GRID_FILE = "grid.npy"
PROFILE_FILE = "profile.json"

# single tile names
ATTRIBUTE_FILE = "attributes.json"
DATASET_FILE = "dataset"
DATAFRAME_FILE = "dataframe.csv"
CARSDICT_FILE = "cars_dict"

PROFILE = "profile"
WINDOW = "window"
OVERLAPS = "overlaps"
ATTRIBUTES = "attributes"
SAVING_INFO = "saving_info"


class CarsDataset:
    """
    CarsDataset.

    Internal CARS structure for organazing tiles
    (xr.Datasets or pd.DataFrames).
    """

    def __init__(self, dataset_type, load_from_disk=None):
        """
        Init function of CarsDataset.
        If a path is provided, restore CarsDataset saved on disk.

        :param dataset_type: type of dataset : 'arrays' or 'points'
        :type dataset_type: str
        :param load_from_disk: path to saved CarsDataset
        :type load_from_disk: str

        """

        self.dataset_type = dataset_type
        if dataset_type not in [
            CARS_DS_TYPE_ARRAY,
            CARS_DS_TYPE_POINTS,
            CARS_DS_TYPE_DICT,
        ]:
            raise ValueError("wrong dataset type")

        self.tiles = None
        self.tiles_info = {}
        self._tiling_grid = None
        self.overlaps = None
        self.attributes = {}

        if load_from_disk is not None:
            self.load_cars_dataset_from_disk(load_from_disk)

    def __repr__(self):
        """
        Repr function
        :return: printable self CarsDataset
        """
        return self.custom_print()

    def __str__(self):
        """
        Str function
        :return: printable self CarsDataset
        """
        return self.custom_print()

    def custom_print(self):
        """
        Return string of self
        :return: printable self
        """

        res = str(self.__class__) + ":  \n" "dataset_type: " + str(
            self.dataset_type
        ) + "\n" + "shape: " + str(self.shape) + "\n" + "tiling_grid: " + str(
            self._tiling_grid
        ) + "\n" + "overlaps: " + str(
            self.overlaps
        ) + "\n" + "tiles_info: " + str(
            self.tiles_info
        ) + "\n" + "attributes: " + str(
            self.attributes
        ) + "\n" + "tiles:" + str(
            self.tiles
        )
        return res

    @property
    def shape(self):
        """
        Return the shape of tiling grid (nb_row, nb_col)
        :return: shape of grid
        """
        return self.tiling_grid.shape[0], self.tiling_grid.shape[1]

    @property
    def tiling_grid(self):
        """
        Tiling grid, containing pixel windows of tiles

        :return: tiling grid, of shape [N, M, 4],
                 containing [row_min, row_max, col_min, col_max]
        :rtype: np.ndarray
        """
        return self._tiling_grid

    @tiling_grid.setter
    def tiling_grid(self, new_grid):
        """
        Set tiling_grid

        :param new_grid: new grid
        :type new_grid: np.ndarray
        """
        self._tiling_grid = new_grid
        # reset overlaps to zeros
        self.overlaps = np.zeros(new_grid.shape)
        # fill dataset grid with Nones
        self.generate_none_tiles()

    def __getitem__(self, key):
        """
        Get item : return the [row, col] dataset

        :param key: tuple index

        :return: tile
        :rtype: xr.Dataset or pd.DataFrame
        """

        if isinstance(key, (tuple, list)):
            if len(key) == 2:
                res = self.tiles[key[0]][key[1]]
            elif len(key) == 1:
                res = self.tiles[key[0]]
            else:
                raise ValueError("Too many indexes, expected 1 or 2")
        else:
            if isinstance(key, int):
                res = self.tiles[key]
            else:
                raise ValueError("Index type not supported")

        return res

    def __setitem__(self, key, newvalue):
        """
        Set new tile

        :param key: tuple of row and col indexes
        :type key: tuple(int, int)
        :param newvalue: tile to set
        """
        if isinstance(key, (tuple, list)):
            if len(key) == 2:
                self.tiles[key[0]][key[1]] = newvalue
            else:
                raise ValueError("Too many indexes, expected 2")
        else:
            raise ValueError("Index type not supported")

    def load_single_tile(self, tile_path_name: str):
        """
        Load a single tile

        :param tile_path_name: Path of tile to load
        :type tile_path_name: str

        :return: single tile
        :rtype: xarray Dataset or Panda dataframe to file

        """

        functions = {
            CARS_DS_TYPE_ARRAY: load_single_tile_array,
            CARS_DS_TYPE_POINTS: load_single_tile_points,
            CARS_DS_TYPE_DICT: load_single_tile_dict,
        }

        return functions[self.dataset_type](tile_path_name)

    def save_single_tile(self, tile, tile_path_name: str):
        """
        Save xarray Dataset or Panda dataframe to file

        :param tile: tile to save
        :type tile: xr.Dataset or pd.DataFrame
        :param tile_path_name: Path of file to save in
        """

        functions = {
            CARS_DS_TYPE_ARRAY: save_single_tile_array,
            CARS_DS_TYPE_POINTS: save_single_tile_points,
            CARS_DS_TYPE_DICT: save_single_tile_dict,
        }

        return functions[self.dataset_type](tile, tile_path_name)

    def run_save(self, future_result, file_name: str, **kwargs):
        """
        Save future result when arrived

        :param future_result: xarray.Dataset received
        :param file_name: filename to save data to
        """

        functions = {
            CARS_DS_TYPE_ARRAY: run_save_arrays,
            CARS_DS_TYPE_POINTS: run_save_points,
        }

        return functions[self.dataset_type](future_result, file_name, **kwargs)

    def get_window_as_dict(self, row, col, from_terrain=False, resolution=1):
        """
        Get window in pixels for rasterio. Set from_terrain if tiling grid
        was defined in geographic coordinates.

        :param row: row
        :type row: int
        :param col: col
        :type col: int
        :param from_terrain: true if in terrain coordinates
        :type from_terrain: bool
        :param resolution: resolution
        :type resolution: float

        :return: New window :  {
            "row_min" : row_min ,
            "row_max" : row_max
            "col_min" : col_min
            "col_max" : col_max
            }
        :rtype: Dict

        """

        row_min = np.min(self.tiling_grid[:, :, 0])
        col_min = np.min(self.tiling_grid[:, :, 2])
        col_max = np.max(self.tiling_grid[:, :, 3])

        window_arr = np.copy(self.tiling_grid[row, col, :])

        if from_terrain:
            #  row -> y axis : reversed by convention
            window = np.array(
                [
                    col_max - window_arr[3],
                    col_max - window_arr[2],
                    window_arr[0] - row_min,
                    window_arr[1] - row_min,
                ]
            )

        else:
            window = np.array(
                [
                    window_arr[0] - row_min,
                    window_arr[1] - row_min,
                    window_arr[2] - col_min,
                    window_arr[3] - col_min,
                ]
            )

        # normalize with resolution
        window = np.round(window / resolution)

        new_window = {
            "row_min": int(window[0]),
            "row_max": int(window[1]),
            "col_min": int(window[2]),
            "col_max": int(window[3]),
        }

        return new_window

    def create_grid(
        self,
        nb_col: int,
        nb_row: int,
        row_split: int,
        col_split: int,
        row_overlap: int,
        col_overlap: int,
    ):
        """
        Generate grid of positions by splitting [0, nb_row]x[0, nb_col]
        in splits of xsplit x ysplit size

        :param nb_col : number of columns
        :param nb_row : number of lines
        :param col_split: width of splits
        :param row_split: height of splits
        :param col_overlap: overlap to apply on rows
        :param row_overlap: overlap to apply on cols

        """
        nb_col_splits = math.ceil(nb_col / row_split)
        nb_row_splits = math.ceil(nb_row / col_split)

        row_min, row_max = 0, nb_row
        col_min, col_max = 0, nb_col

        out_grid = np.ndarray(
            shape=(nb_row_splits, nb_col_splits, 4), dtype=int
        )

        out_overlap = np.ndarray(
            shape=(nb_row_splits, nb_col_splits, 4), dtype=int
        )

        for i in range(0, nb_row_splits):
            for j in range(0, nb_col_splits):
                row_down = row_min + row_split * i
                col_left = col_min + col_split * j
                row_up = min(row_max, row_min + (i + 1) * row_split)
                col_right = min(col_max, col_min + (j + 1) * col_split)

                out_grid[i, j, 0] = row_down
                out_grid[i, j, 1] = row_up
                out_grid[i, j, 2] = col_left
                out_grid[i, j, 3] = col_right

                # fill overlap [OL_row_down, OL_row_up, OL_col_left,
                #  OL_col_right]
                out_overlap[i, j, 0] = row_down - max(
                    row_min, row_down - row_overlap
                )
                out_overlap[i, j, 1] = (
                    min(row_max, row_up + row_overlap) - row_up
                )
                out_overlap[i, j, 2] = col_left - max(
                    col_min, col_left - col_overlap
                )
                out_overlap[i, j, 3] = (
                    min(col_right, col_right + col_overlap) - col_right
                )

        self.tiling_grid = out_grid
        self.overlaps = out_overlap

    def generate_none_tiles(self):
        """
        Generate the structure of data tiles, with Nones, according
            to grid shape.

        """

        self.tiles = create_none(
            self.tiling_grid.shape[0], self.tiling_grid.shape[1]
        )

    def create_empty_copy(self, cars_ds):
        """
        Copy attributes, grid, overlaps, and create Nones.

        :param cars_ds: CarsDataset to copy
        :type cars_ds: CarsDataset

        """

        self.tiles_info = copy.deepcopy(cars_ds.tiles_info)
        self.tiling_grid = copy.deepcopy(cars_ds.tiling_grid)
        self.overlaps = copy.deepcopy(cars_ds.overlaps)

        self.tiles = []
        for _ in range(cars_ds.overlaps.shape[0]):
            tiles_row = []
            for _ in range(cars_ds.overlaps.shape[1]):
                tiles_row.append(None)
            self.tiles.append(tiles_row)

    def generate_descriptor(
        self, future_result, file_name, tag=None, dtype=None, nodata=None
    ):
        """
        Generate de rasterio descriptor for the given future result

        Only works with pixelic tiling grid

        :param future_result: Future result
        :type future_result: xr.Dataset
        :param file_name: file name to save futures to
        :type file_name: str
        :param tag: tag to save
        :type tag: str
        :param dtype: dtype
        :type dtype: str
        :param nodata: no data value
        :type nodata: float
        """

        # Get profile from 1st finished future
        new_profile = get_profile_for_tag_dataset(future_result, tag)

        if "width" not in new_profile or "height" not in new_profile:
            logging.debug(
                "CarsDataset doesn't have a profile, default is given"
            )
            new_profile = DefaultGTiffProfile(count=new_profile["count"])
            new_profile["height"] = np.max(self.tiling_grid[:, :, 1])
            new_profile["width"] = np.max(self.tiling_grid[:, :, 3])

        # Change dtype
        new_profile["dtype"] = dtype
        if nodata is not None:
            new_profile["nodata"] = nodata

        descriptor = rio.open(file_name, "w", **new_profile, BIGTIFF="IF_SAFER")

        return descriptor

    def save_cars_dataset(self, directory):
        """
        Save whole CarsDataset to given directory, including tiling grids,
        attributes, overlaps, and all the xr.Dataset or pd.DataFrames.

        :param directory: Path where to save  self CarsDataset
        :type directory: str

        """

        # Create CarsDataset folder
        safe_makedirs(directory)

        if self.tiles is None:
            logging.error("No tiles managed by CarsDatasets")
            raise RuntimeError("No tiles managed by CarsDatasets")

        # save tiles info
        tiles_info_file = os.path.join(directory, TILES_INFO_FILE)
        save_dict(self.tiles_info, tiles_info_file)

        # save grid
        grid_file = os.path.join(directory, GRID_FILE)
        save_numpy_array(self.tiling_grid, grid_file)

        # save overlap
        overlap_file = os.path.join(directory, OVERLAP_FILE)
        save_numpy_array(self.overlaps, overlap_file)

        nb_rows, nb_cols = self.tiling_grid.shape[0], self.tiling_grid.shape[1]

        # save each tile
        for col in range(nb_cols):
            for row in range(nb_rows):
                # Get name
                current_tile_path_name = create_tile_path(col, row, directory)

                # save tile
                self.save_single_tile(
                    self.tiles[row][col], current_tile_path_name
                )

    def load_cars_dataset_from_disk(self, directory):
        """
        Load whole CarsDataset from given directory

        :param directory: Path where is saved CarsDataset to load
        :type directory: str

        """

        # get tiles info
        tiles_info_file = os.path.join(directory, TILES_INFO_FILE)
        self.tiles_info = load_dict(tiles_info_file)

        # load grid
        grid_file = os.path.join(directory, GRID_FILE)
        self.tiling_grid = load_numpy_array(grid_file)

        nb_rows, nb_cols = self.tiling_grid.shape[0], self.tiling_grid.shape[1]

        # load overlap
        overlap_file = os.path.join(directory, OVERLAP_FILE)
        self.overlaps = load_numpy_array(overlap_file)

        # load each tile
        self.tiles = []
        for row in range(nb_rows):
            tiles_row = []
            for col in range(nb_cols):
                # Get name
                current_tile_path_name = create_tile_path(col, row, directory)

                # load tile
                tiles_row.append(self.load_single_tile(current_tile_path_name))

            self.tiles.append(tiles_row)


def run_save_arrays(future_result, file_name, tag=None, descriptor=None):
    """
    Save future when arrived

    :param future_result: xarray.Dataset received
    :type future_result: xarray.Dataset
    :param file_name: filename to save data to
    :type file_name: str
    :param tag: dataset tag to rasterize
    :type tag: str
    :param descriptor: rasterio descriptor
    """
    # write future result using saved window and overlaps

    save_dataset(
        future_result,
        file_name,
        tag,
        use_windows_and_overlaps=True,
        descriptor=descriptor,
    )


def run_save_points(future_result, file_name, overwrite=False):
    """
    Save future result when arrived

    :param future_result: pandas Dataframe received
    :type future_result: pandas Dataframe
    :param file_name: filename to save data to
    :type file_name: str
    :param overwrite: overwrite file
    :type overwrite: bool

    """

    if overwrite:
        # remove pickle file if already exists
        if os.path.exists(file_name):
            os.remove(file_name)
    # Save
    save_dataframe(future_result, file_name, overwrite=False)


def load_single_tile_array(tile_path_name: str) -> xr.Dataset:
    """
    Load a xarray tile

    :param tile_path_name: Path of tile to load
    :type tile_path_name: str

    :return: tile dataset
    :rtype: xr.Dataset

    """

    # get dataset
    dataset_file_name = os.path.join(tile_path_name, DATASET_FILE)
    with open(dataset_file_name, "rb") as handle:
        dataset = pickle.load(handle)

    # get attributes
    attributes_file_name = os.path.join(tile_path_name, ATTRIBUTE_FILE)
    attributes = load_dict(attributes_file_name)

    # Format transformation
    if PROFILE in attributes:
        attributes[PROFILE] = dict_profile_to_rio_profile(attributes[PROFILE])

    # add to dataset
    dataset.attrs.update(attributes)

    return dataset


def load_single_tile_points(tile_path_name: str):
    """
    Load a panda dataframe

    :param tile_path_name: Path of tile to load
    :type tile_path_name: str

    :return: Tile dataframe
    :rtype: Panda dataframe

    """

    # get dataframe
    dataframe_file_name = os.path.join(tile_path_name, DATAFRAME_FILE)
    with open(dataframe_file_name, "rb") as handle:
        dataframe = pickle.load(handle)

    # get attributes
    attributes_file_name = os.path.join(tile_path_name, ATTRIBUTE_FILE)
    attributes = load_dict(attributes_file_name)

    # Format transformation

    # add to dataframe
    dataframe.attrs.update(attributes)

    return dataframe


def load_single_tile_dict(tile_path_name: str):
    """
    Load a CarsDict

    :param tile_path_name: Path of tile to load
    :type tile_path_name: str

    :return: Tile dataframe
    :rtype: Panda dataframe

    """

    # get dataframe
    dict_file_name = os.path.join(tile_path_name, CARSDICT_FILE)
    with open(dict_file_name, "rb") as handle:
        dict_cars = pickle.load(handle)

    # get attributes
    attributes_file_name = os.path.join(tile_path_name, ATTRIBUTE_FILE)
    attributes = load_dict(attributes_file_name)

    # Format transformation

    # add to dataframe
    dict_cars.attrs.update(attributes)

    return dict_cars


def save_single_tile_array(dataset: xr.Dataset, tile_path_name: str):
    """
    Save xarray to directory, saving the data in a different file that
    the attributes (saved in a .json next to it).

    :param dataset: dataset to save
    :type dataset: xr.Dataset
    :param tile_path_name: Path of file to save in
    :type tile_path_name: str
    """
    # Create tile folder
    safe_makedirs(tile_path_name)

    # save attributes
    saved_dataset_attrs = copy.copy(dataset.attrs)
    attributes_file_name = os.path.join(tile_path_name, ATTRIBUTE_FILE)
    if dataset.attrs is None:
        attributes = {}
    else:
        attributes = dataset.attrs

    # Format transformation
    if PROFILE in attributes:
        attributes[PROFILE] = rio_profile_to_dict_profile(attributes[PROFILE])

    # dump
    # separate attributes
    dataset.attrs, custom_attributes = separate_dicts(
        attributes, [PROFILE, WINDOW, OVERLAPS, SAVING_INFO, ATTRIBUTES]
    )
    # save
    save_dict(custom_attributes, attributes_file_name)
    dataset_file_name = os.path.join(tile_path_name, DATASET_FILE)
    with open(dataset_file_name, "wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Retrieve attrs
    dataset.attrs = saved_dataset_attrs


def save_single_tile_points(dataframe, tile_path_name: str):
    """
    Save dataFrame to directory, saving the data in a different file that
    the attributes (saved in a .json next to it).

    :param dataframe: dataframe to save
    :type dataframe: pd.DataFrame
    :param tile_path_name: Path of file to save in
    :type tile_path_name: str
    """
    # Create tile folder
    safe_makedirs(tile_path_name)

    # save attributes
    saved_dataframe_attrs = copy.copy(dataframe.attrs)
    attributes_file_name = os.path.join(tile_path_name, ATTRIBUTE_FILE)
    if dataframe.attrs is None:
        attributes = {}
    else:
        attributes = dataframe.attrs

    # Format transformation

    # dump
    # separate attributes
    dataframe.attrs, custom_attributes = separate_dicts(
        attributes, [SAVING_INFO, ATTRIBUTES]
    )
    # save
    save_dict(custom_attributes, attributes_file_name)
    dataframe_file_name = os.path.join(tile_path_name, DATAFRAME_FILE)
    with open(dataframe_file_name, "wb") as handle:
        pickle.dump(dataframe, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Retrieve attrs
    dataframe.attrs = saved_dataframe_attrs


def save_single_tile_dict(dict_cars, tile_path_name: str):
    """
    Save cars_dict to directory, saving the data in a different file that
    the attributes (saved in a .json next to it).

    :param dict_cars: dataframe to save
    :type dict_cars: pd.DataFrame
    :param tile_path_name: Path of file to save in
    :type tile_path_name: str
    """
    # Create tile folder
    safe_makedirs(tile_path_name)

    # save attributes
    saved_dict_cars_attrs = copy.copy(dict_cars.attrs)
    attributes_file_name = os.path.join(tile_path_name, ATTRIBUTE_FILE)
    if dict_cars.attrs is None:
        attributes = {}
    else:
        attributes = dict_cars.attrs

    # Format transformation

    # dump
    # separate attributes
    dict_cars.attrs, custom_attributes = separate_dicts(
        attributes, [SAVING_INFO, ATTRIBUTES]
    )
    # save
    save_dict(custom_attributes, attributes_file_name)
    dict_cars_file_name = os.path.join(tile_path_name, CARSDICT_FILE)
    with open(dict_cars_file_name, "wb") as handle:
        pickle.dump(dict_cars, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Retrieve attrs
    dict_cars.attrs = saved_dict_cars_attrs


def fill_dataset(
    dataset,
    saving_info=None,
    window=None,
    profile=None,
    attributes=None,
    overlaps=None,
):
    """
    From a full xarray dataset, fill info properly.
    User can fill with saving information (containing CarsDataset id),
    window of current tile and its overlaps,
    rasterio profile of full data, and attributes associated to data

    :param dataset: dataset to fill
    :type dataset: xarray_dataset
    :param saving_info: created by Orchestrator.get_saving_infos
    :type saving_info: dict
    :param window:
    :type window: dict
    :param profile:
    :type profile: dict
    :param attributes:
    :type attributes: dict

    """

    if attributes is not None:
        dataset.attrs[ATTRIBUTES] = attributes

    if saving_info is not None:
        dataset.attrs[SAVING_INFO] = saving_info

    if window is not None:
        dataset.attrs[WINDOW] = window

    if overlaps is not None:
        dataset.attrs[OVERLAPS] = overlaps

    if profile is not None:
        dataset.attrs[PROFILE] = profile


def fill_dataframe(dataframe, saving_info=None, attributes=None):
    """
    From a full pandas dataframe, fill info properly.
    User can fill with saving information (containing CarsDataset id),
    and attributes associated to data


    :param dataframe: dataframe to fill
    :type dataframe: pandas dataframe
    :param saving_info: created by Orchestrator.get_saving_infos
    :type saving_info: dict
    :param attributes:
    :type attributes: dict

    """

    if attributes is not None:
        dataframe.attrs[ATTRIBUTES] = attributes

    if saving_info is not None:
        dataframe.attrs[SAVING_INFO] = saving_info


def fill_dict(data_dict, saving_info=None, attributes=None):
    """
    From a fulldict, fill info properly.
    User can fill with saving information (containing CarsDataset id),
    and attributes associated to data


    :param data_dict: dictionnary to fill
    :type data_dict: Dict
    :param saving_info: created by Orchestrator.get_saving_infos
    :type saving_info: dict
    :param attributes:
    :type attributes: dict

    """

    # TODO only use CarsDict

    if isinstance(data_dict, dict):
        if attributes is not None:
            data_dict[ATTRIBUTES] = attributes

        if saving_info is not None:
            data_dict[SAVING_INFO] = saving_info

    elif isinstance(data_dict, cars_dict.CarsDict):
        if attributes is not None:
            data_dict.attrs[ATTRIBUTES] = attributes

        if saving_info is not None:
            data_dict.attrs[SAVING_INFO] = saving_info


def save_dataframe(dataframe, file_name, overwrite=True):
    """
    Save DataFrame to csv format. The content of dataframe is merged to
    the content of existing saved Dataframe, if overwrite==False

    :param file_name: file name to save data to
    :type file_name: str
    :param overwrite: overwrite file if exists
    :type overwrite: bool

    """
    # generate filename if attributes have xstart and ystart settings
    if (
        "attributes" in dataframe.attrs
        and "xmin" in dataframe.attrs["attributes"]
    ):
        file_name = os.path.dirname(file_name)
        file_name = os.path.join(
            file_name,
            (
                str(dataframe.attrs["attributes"]["xmin"])
                + "_"
                + str(dataframe.attrs["attributes"]["ymax"])
            ),
        )

    # Save attributes
    attributes_file_name = file_name + "_attrs.json"
    save_dict(dataframe.attrs, attributes_file_name)

    # Save point cloud to laz format
    if (
        "attributes" in dataframe.attrs
        and "save_points_cloud_as_laz" in dataframe.attrs["attributes"]
    ):
        if dataframe.attrs["attributes"]["save_points_cloud_as_laz"]:
            las_file_name = file_name + ".laz"
            dataframe_converter.convert_pcl_to_laz(dataframe, las_file_name)

    # Save panda dataframe to csv
    if (
        (
            "attributes" in dataframe.attrs
            and "save_points_cloud_as_csv" in dataframe.attrs["attributes"]
            and dataframe.attrs["attributes"]["save_points_cloud_as_csv"]
        )
        or "attributes" not in dataframe.attrs
        or "save_points_cloud_as_csv" not in dataframe.attrs["attributes"]
    ):
        _, extension = os.path.splitext(file_name)
        if "csv" not in extension:
            file_name = file_name + ".csv"
        if overwrite and os.path.exists(file_name):
            dataframe.to_csv(file_name, index=False)
        else:
            if os.path.exists(file_name):
                # merge files
                existing_dataframe = pandas.read_csv(file_name)
                merged_dataframe = pandas.concat(
                    [existing_dataframe, dataframe],
                    ignore_index=True,
                    sort=False,
                )
                merged_dataframe.to_csv(file_name, index=False)

            else:
                dataframe.to_csv(file_name, index=False)


def save_dataset(
    dataset, file_name, tag, use_windows_and_overlaps=False, descriptor=None
):
    """
    Reconstruct and save data.
    In order to save properly the dataset to corresponding tiff file,
    dataset must have been filled with saving info, profile, window,
    overlaps (if not 0), and rasterio descriptor if already created.
    See fill_dataset.

    :param dataset: dataset to save
    :type dataset: xr.Dataset
    :param file_name: file name to save data to
    :type file_name: str
    :param tag: tag to reconstruct
    :type tag: str
    :param use_windows_and_overlaps: use saved window and overlaps
    :type use_windows_and_overlaps: bool
    :param descriptor: descriptor to use with rasterio
    :type descriptor: rasterio dataset

    """
    overlaps = get_overlaps_dataset(dataset)
    window = get_window_dataset(dataset)

    rio_window = None
    overlap = [0, 0, 0, 0]
    if use_windows_and_overlaps:
        if window is None:
            logging.debug("User wants to use window but none was set")

        else:
            rio_window = generate_rasterio_window(window)

            if overlaps is not None:
                overlap = [
                    overlaps["up"],
                    overlaps["down"],
                    overlaps["left"],
                    overlaps["right"],
                ]
    if len(dataset[tag].values.shape) > 2:
        nb_rows, nb_cols = (
            dataset[tag].values.shape[1],
            dataset[tag].values.shape[2],
        )

        data = dataset[tag].values[
            :,
            overlap[0] : nb_rows - overlap[1],
            overlap[2] : nb_cols - overlap[3],
        ]
    else:
        nb_rows, nb_cols = (
            dataset[tag].values.shape[0],
            dataset[tag].values.shape[1],
        )

        data = dataset[tag].values[
            overlap[0] : nb_rows - overlap[1],
            overlap[2] : nb_cols - overlap[3],
        ]

    if tag == cst.EPI_COLOR and "int" in descriptor.dtypes[0]:
        # Prepare color data for cast
        data = np.nan_to_num(data, nan=descriptor.nodata)
        data = np.round(data)

    profile = get_profile_for_tag_dataset(dataset, tag)

    new_profile = profile
    if "width" not in new_profile or "height" not in new_profile:
        logging.debug("CarsDataset doesn't have a profile, default is given")
        new_profile = DefaultGTiffProfile(count=new_profile["count"])
        new_profile["height"] = data.shape[0]
        new_profile["width"] = data.shape[1]
        new_profile["dtype"] = "float32"

    bands_description = None
    if tag in (cst.EPI_CLASSIFICATION, cst.RASTER_CLASSIF):
        bands_description = dataset.coords[cst.BAND_CLASSIF].values
    if tag in (cst.EPI_COLOR, cst.POINTS_CLOUD_CLR_KEY_ROOT):
        bands_description = dataset.coords[cst.BAND_IM].values
    if tag == cst.RASTER_SOURCE_PC:
        bands_description = dataset.coords[cst.BAND_SOURCE_PC].values

    outputs.rasterio_write_georaster(
        file_name,
        data,
        new_profile,
        window=rio_window,
        descriptor=descriptor,
        bands_description=bands_description,
    )


def create_tile_path(col: int, row: int, directory: str) -> str:
    """
    Create path of tile, according to its position in CarsDataset grid

    :param col: numero of column
    :type col: int
    :param row: numero of row
    :type row: int
    :param directory: path where to save tile
    :type directory: str

    :return: full path
    :rtype: str

    """

    tail = "col_" + repr(col) + "_row_" + repr(row)
    name = os.path.join(directory, tail)

    return name


def save_numpy_array(array: np.ndarray, file_name: str):
    """
    Save numpy array to file

    :param array: array to save
    :type array: np.ndarray
    :param file_name: numero of row
    :type file_name: str

    """

    with open(file_name, "wb") as descriptor:
        np.save(descriptor, array)


def load_numpy_array(file_name: str) -> np.ndarray:
    """
    Load numpy array from file

    :param file_name: numero of row
    :type file_name: str

    :return: array
    :rtype: np.ndarray

    """
    with open(file_name, "rb") as descriptor:
        return np.load(descriptor)


def create_none(nb_row: int, nb_col: int):
    """
    Create a grid filled with None. The created grid is a 2D list :
    ex: [[None, None], [None, None]]

    :param nb_row: number of rows
    :param nb_col: number of cols
    :return: Grid filled with None
    :rtype: list of list
    """
    grid = []
    for _ in range(nb_row):
        tmp = []
        for _ in range(nb_col):
            tmp.append(None)
        grid.append(tmp)
    return grid


def overlap_array_to_dict(overlap):
    """
    Convert matrix of overlaps, to dict format used in CarsDatasets.
    Input is : [o_up, o_down, o_left, o_right].
    Output is : {"up": o_up, "down": o_down, "left": o_left, "right": o_right}

    :param overlap: overlaps
    :type overlap: List

    :return: New overlaps
    :rtype: Dict

    """
    new_overlap = {
        "up": int(overlap[0]),
        "down": int(overlap[1]),
        "left": int(overlap[2]),
        "right": int(overlap[3]),
    }
    return new_overlap


def window_array_to_dict(window, overlap=None):
    """
    Convert matrix of windows, to dict format used in CarsDatasets.
    Use overlaps if you want to get window with overlaps
    inputs are :

      - window : [row_min, row_max, col_min, col_max], with pixel format
      - overlap (optional): [o_row_min, o_row_max, o_col_min, o_col_max]

    outputs are :
      {
          "row_min" : row_min - o_row_min,
          "row_max" : row_max + o_row_max,
          "col_min" : col_min - o_col_min,
          "col_max" : col_max - o_col_max,

      }

    :param window: window
    :type window: List
    :param overlap: overlaps
    :type overlap: List

    :return: New window
    :rtype: Dict

    """

    new_window = {
        "row_min": int(window[0]),
        "row_max": int(window[1]),
        "col_min": int(window[2]),
        "col_max": int(window[3]),
    }

    if overlap is not None:
        new_window["row_min"] -= int(overlap[0])
        new_window["row_max"] += int(overlap[1])
        new_window["col_min"] -= int(overlap[2])
        new_window["col_max"] += int(overlap[3])

    return new_window


def dict_profile_to_rio_profile(dict_profile: Dict) -> Dict:
    """
    Transform a rasterio Profile transformed into serializable Dict,
    into a rasterio profile.

    :param profile: rasterio Profile transformed into serializable Dict
    :type profile: Dict

    :return: Profile
    :rtype: Rasterio Profile

    """

    rio_profile = copy.copy(dict_profile)

    transform = None
    if "transform" in dict_profile:
        if dict_profile["transform"] is not None:
            transform = rio.Affine(
                dict_profile["transform"][0],
                dict_profile["transform"][1],
                dict_profile["transform"][2],
                dict_profile["transform"][3],
                dict_profile["transform"][4],
                dict_profile["transform"][5],
            )
    crs = None
    if "crs" in dict_profile:
        if dict_profile["crs"] is not None:
            if isinstance(dict_profile["crs"], str):
                crs = rio.crs.CRS.from_epsg(
                    dict_profile["crs"].replace("EPSG:", "")
                )
            else:
                crs = rio.crs.CRS.from_epsg(dict_profile["crs"])

    rio_profile["crs"] = crs
    rio_profile["transform"] = transform

    return rio_profile


def rio_profile_to_dict_profile(in_profile: Dict) -> Dict:
    """
    Transform a rasterio profile into a serializable Dict.

    :param in_profile: rasterio Profile transformed into serializable Dict
    :type in_profile: Dict

    :return: Profile
    :rtype: Dict

    """

    profile = copy.copy(in_profile)

    profile = {**profile}
    crs = None
    if "crs" in profile:
        if profile["crs"] is not None:
            if isinstance(profile["crs"], str):
                crs = profile["crs"]
            else:
                crs = profile["crs"].to_epsg()

    transform = None
    if "transform" in profile:
        if profile["transform"] is not None:
            transform = list(profile["transform"])[:6]

    profile.update(crs=crs, transform=transform)

    return profile


def save_dict(dictionary, file_path: str, safe_save=False):
    """
    Save dict to json file

    :param dictionary: dictionary to save
    :type dictionary: Dict
    :param file_path: file path to use
    :type file_path: str
    :param safe_save: if True, be robust to types
    :type safe_save: bool

    """

    class CustomEncoder(json.JSONEncoder):
        """
        Custom json encoder

        """

        def default(self, o):
            """
            Converter
            """
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return json.JSONEncoder.default(self, o)

    if safe_save:
        with open(file_path, "w", encoding="utf8") as fstream:
            json.dump(dictionary, fstream, indent=2, cls=CustomEncoder)
    else:
        with open(file_path, "w", encoding="utf8") as fstream:
            json.dump(dictionary, fstream, indent=2)


def load_dict(file_path: str) -> Dict:
    """
    Load dict from json file

    :param file_path: file path to use
    :type file_path: str

    """

    with open(file_path, "r", encoding="utf8") as fstream:
        dictionary = json.load(fstream)

    return dictionary


def separate_dicts(dictionary, list_tags):
    """
    Separate a dict into two, the second one containing the given tags.

    For example, {key1: val1, key2: val2, key3: val3}
    with list_tags = [key2] will be split in :
    {key1: val1, key3: val3} and {key2: val2}

    """

    dict1 = {}
    dict2 = {}

    for key in dictionary:
        if key in list_tags:
            dict2[key] = dictionary[key]
        else:
            dict1[key] = dictionary[key]

    return dict1, dict2


def get_attributes_dataframe(dataframe):
    """
    Get attributes field in .attr of dataframe

    :param dataframe: dataframe
    :type dataframe: pandas dataframe
    """

    return dataframe.attrs.get(ATTRIBUTES, None)


def get_window_dataset(dataset):
    """
    Get window in dataset

    :param dataset: dataset
    :type dataset: xr.Dataset
    """

    return dataset.attrs.get(WINDOW, None)


def get_overlaps_dataset(dataset):
    """
    Get overlaps in dataset

    :param dataset: dataset
    :type dataset: xr.Dataset
    """

    return dataset.attrs.get(OVERLAPS, None)


def get_profile_rasterio(dataset):
    """
    Get profile in dataset

    :param dataset: dataset
    :type dataset: xr.Dataset
    """

    return dataset.attrs.get(PROFILE, None)


def get_attributes(dataset):
    """
    Get attributes in dataset

    :param dataset: dataset
    :type dataset: xr.Dataset
    """

    return dataset.attrs.get(ATTRIBUTES, None)


def get_profile_for_tag_dataset(dataset, tag: str) -> Dict:
    """
    Get profile according to layer to save.
    This function modify current rasterio dataset to fix the number of
    bands of the data associated to given tag.

    :param tag: tag to use
    :type tag: str

    :return: Profile
    :rtype: Rasterio Profile

    """

    new_profile = get_profile_rasterio(dataset)
    if new_profile is None:
        new_profile = {}

    new_profile["count"] = 1
    if len(dataset[tag].values.shape) > 2:
        new_profile["count"] = dataset[tag].values.shape[0]

    return new_profile


def generate_rasterio_window(window: Dict) -> rio.windows.Window:
    """
    Generate rasterio window to use.

    :param window: window to convert, containing 'row_min',
                'row_max', 'col_min', 'col_max
    :type window: dict

    :return: rasterio window
    :rtype: rio.windows.Window

    """
    returned_window = None

    if window is not None:
        return Window.from_slices(
            (window["row_min"], window["row_max"]),
            (window["col_min"], window["col_max"]),
        )

    return returned_window
