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
this module contains the abstract geometry class to use in the
geometry plugins
"""
import logging
import struct
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import rasterio as rio
import xarray as xr
from scipy import interpolate
from shapely.geometry import Polygon

from cars.conf import input_parameters
from cars.core import constants as cst
from cars.core import inputs, outputs
from cars.data_structures import cars_dataset


class AbstractGeometry(metaclass=ABCMeta):
    """
    AbstractGeometry
    """

    available_loaders: Dict = {}

    def __new__(cls, loader_to_use):
        """
        Return the required loader
        :raises:
         - KeyError when the required loader is not registered

        :param loader_to_use: loader name to instantiate
        :return: a loader_to_use object
        """

        if loader_to_use not in cls.available_loaders:
            logging.error(
                "No geometry loader named {} registered".format(loader_to_use)
            )
            raise KeyError(
                "No geometry loader named {} registered".format(loader_to_use)
            )

        logging.info(
            "The AbstractGeometry {} loader will be used".format(loader_to_use)
        )

        return super(AbstractGeometry, cls).__new__(
            cls.available_loaders[loader_to_use]
        )

    @classmethod
    def register_subclass(cls, short_name: str):
        """
        Allows to register the subclass with its short name
        :param short_name: the subclass to be registered
        :type short_name: string
        """

        def decorator(subclass):
            """
            Registers the subclass in the available methods
            :param subclass: the subclass to be registered
            :type subclass: object
            """
            cls.available_loaders[short_name] = subclass
            return subclass

        return decorator

    @property
    @abstractmethod
    def conf_schema(self):
        """
        Returns the input configuration fields required by the geometry loader
        as a json checker schema. The available fields are defined in the
        cars/conf/input_parameters.py file

        :return: the geo configuration schema
        """

    @staticmethod
    @abstractmethod
    def triangulate(
        cars_conf,
        mode: str,
        matches: Union[xr.Dataset, np.ndarray],
        grid1: str,
        grid2: str,
        roi_key: Union[None, str] = None,
    ) -> np.ndarray:
        """
        Performs triangulation from cars disparity or matches dataset

        :param cars_conf: cars input configuration dictionary
        :param mode: triangulation mode
               (constants.DISP_MODE or constants.MATCHES)
        :param matches: cars disparity dataset or matches as numpy array
        :param grid1: path to epipolar grid of img1
        :param grid2: path to epipolar grid of image 2
        :param roi_key: dataset roi to use
               (can be cst.ROI or cst.ROI_WITH_MARGINS)
        :return: the long/lat/height numpy array in output of the triangulation
        """

    @staticmethod
    @abstractmethod
    def check_products_consistency(cars_conf) -> bool:
        """
        Test if the product is readable by the geometry loader

        :param: cars_conf: cars input configuration dictionary
        :return: True if the products are readable, False otherwise
        """

    @staticmethod
    @abstractmethod
    def generate_epipolar_grids(
        cars_conf,
        dem: Union[None, str] = None,
        geoid: Union[None, str] = None,
        default_alt: Union[None, float] = None,
        epipolar_step: int = 30,
    ) -> Tuple[
        np.ndarray, np.ndarray, List[float], List[float], List[int], float
    ]:
        """
        Computes the left and right epipolar grids

        :param cars_conf: cars input configuration dictionary
        :param dem: path to the dem folder
        :param geoid: path to the geoid file
        :param default_alt: default altitude to use in the missing dem regions
        :param epipolar_step: step to use to construct the epipolar grids
        :return: Tuple composed of :

            - the left epipolar grid as a numpy array
            - the right epipolar grid as a numpy array
            - the left grid origin as a list of float
            - the left grid spacing as a list of float
            - the epipolar image size as a list of int \
            (x-axis size is given with the index 0, y-axis size with index 1)
            - the disparity to altitude ratio as a float
        """

    @staticmethod
    def matches_to_sensor_coords(
        grid1: Union[str, cars_dataset.CarsDataset],
        grid2: Union[str, cars_dataset.CarsDataset],
        matches: np.ndarray,
        matches_type: str,
        matches_msk: np.ndarray = None,
        ul_matches_shift: Tuple[int, int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert matches (sparse or dense matches) given in epipolar
        coordinates to sensor coordinates. This function is available for
        loaders if it requires matches in sensor coordinates to perform
        the triangulation.

        This function returns a tuple composed of the matches left and right
        sensor coordinates as numpy arrays. For each original image, the sensor
        coordinates are arranged as follows :

            - if the matches are a vector of matching points: a numpy array of\
            size [number of matches, 2].\
            The last index indicates the 'x' coordinate(last index set to 0) or\
            the 'y' coordinate (last index set to 1).
            - if matches is a cars disparity dataset: a numpy array of size \
            [nb_epipolar_line, nb_epipolar_col, 2]. Where\
            [nb_epipolar_line, nb_epipolar_col] is the size of the disparity \
            map. The last index indicates the 'x' coordinate (last index set \
            to 0) or the 'y' coordinate (last index set to 1).

        :param grid1: path to epipolar grid of image 1
        :param grid2: path to epipolar grid of image 2
        :param matches: cars disparity dataset or matches as numpy array
        :param matches_type: matches type (cst.DISP_MODE or cst.MATCHES)
        :param matches_msk: matches mask to provide for cst.DISP_MODE
        :param ul_matches_shift: coordinates (x, y) of the upper left corner of
               the matches map (for cst.DISP_MODE) in the original epipolar
               geometry (use this if the map have been cropped)
        :return: a tuple of numpy array. The first array corresponds to the
                 left matches in sensor coordinates, the second one is the right
                 matches in sensor coordinates.
        """
        vec_epi_pos_left = None
        vec_epi_pos_right = None

        if matches_type == cst.MATCHES_MODE:
            # retrieve left and right matches
            vec_epi_pos_left = matches[:, 0:2]
            vec_epi_pos_right = matches[:, 2:4]

        elif matches_type == cst.DISP_MODE:
            if matches_msk is None:
                logging.error("No disparity mask given in input")
                raise RuntimeError("No disparity mask given in input")

            if ul_matches_shift is None:
                ul_matches_shift = (0, 0)

            # convert disparity to matches
            epi_pos_left_y, epi_pos_left_x = np.mgrid[
                ul_matches_shift[1] : ul_matches_shift[1] + matches.shape[0],
                ul_matches_shift[0] : ul_matches_shift[0] + matches.shape[1],
            ]

            epi_pos_left_x = epi_pos_left_x.astype(np.float64)
            epi_pos_left_y = epi_pos_left_y.astype(np.float64)
            epi_pos_right_y = np.copy(epi_pos_left_y)
            epi_pos_right_x = np.copy(epi_pos_left_x)
            epi_pos_right_x[np.where(matches_msk == 255)] += matches[
                np.where(matches_msk == 255)
            ]

            # vectorize matches
            vec_epi_pos_left = np.transpose(
                np.vstack([epi_pos_left_x.ravel(), epi_pos_left_y.ravel()])
            )
            vec_epi_pos_right = np.transpose(
                np.vstack([epi_pos_right_x.ravel(), epi_pos_right_y.ravel()])
            )

        # convert epipolar matches to sensor coordinates
        sensor_pos_left = AbstractGeometry.sensor_position_from_grid(
            grid1, vec_epi_pos_left
        )
        sensor_pos_right = AbstractGeometry.sensor_position_from_grid(
            grid2, vec_epi_pos_right
        )

        if matches_type == cst.DISP_MODE:
            # rearrange matches in the original epipolar geometry
            sensor_pos_left_x = sensor_pos_left[:, 0].reshape(matches_msk.shape)
            sensor_pos_left_x[np.where(matches_msk != 255)] = np.nan
            sensor_pos_left_y = sensor_pos_left[:, 1].reshape(matches_msk.shape)
            sensor_pos_left_y[np.where(matches_msk != 255)] = np.nan

            sensor_pos_right_x = sensor_pos_right[:, 0].reshape(
                matches_msk.shape
            )
            sensor_pos_right_x[np.where(matches_msk != 255)] = np.nan
            sensor_pos_right_y = sensor_pos_right[:, 1].reshape(
                matches_msk.shape
            )
            sensor_pos_right_y[np.where(matches_msk != 255)] = np.nan

            sensor_pos_left = np.zeros(
                (matches_msk.shape[0], matches_msk.shape[1], 2)
            )
            sensor_pos_left[:, :, 0] = sensor_pos_left_x
            sensor_pos_left[:, :, 1] = sensor_pos_left_y
            sensor_pos_right = np.zeros(
                (matches_msk.shape[0], matches_msk.shape[1], 2)
            )
            sensor_pos_right[:, :, 0] = sensor_pos_right_x
            sensor_pos_right[:, :, 1] = sensor_pos_right_y

        return sensor_pos_left, sensor_pos_right

    @staticmethod
    def sensor_position_from_grid(
        grid: Union[str, cars_dataset.CarsDataset],
        positions: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate the positions given as inputs using the grid

        :param grid: path to epipolar grid, or numpy array
        :param positions: epipolar positions to interpolate given as a numpy
               array of size [number of points, 2]. The last index indicates
               the 'x' coordinate (last index set to 0) or the 'y' coordinate
               (last index set to 1).
        :return: sensors positions as a numpy array of size
                 [number of points, 2]. The last index indicates the 'x'
                 coordinate (last index set to 0) or
                 the 'y' coordinate (last index set to 1).
        """

        if isinstance(grid, str):
            # open epipolar grid
            ds_grid = rio.open(grid)

            # retrieve grid step
            transform = ds_grid.transform
            step_col = transform[0]
            step_row = transform[4]

            # center-pixel positions
            [ori_col, ori_row] = transform * (0.5, 0.5)

            last_col = ori_col + step_col * ds_grid.width
            last_row = ori_row + step_row * ds_grid.height

            # transform dep to positions
            row_dep = ds_grid.read(2)
            col_dep = ds_grid.read(1)

        elif isinstance(grid, cars_dataset.CarsDataset):
            # Get data
            grid_data = grid[0, 0]
            row_dep = grid_data[:, :, 1]
            col_dep = grid_data[:, :, 0]

            # Get step
            step_col = grid.attributes["grid_spacing"][1]
            step_row = grid.attributes["grid_spacing"][0]
            ori_col = step_col / 2
            ori_row = step_row / 2
            last_row = ori_row + step_row * grid_data.shape[0]
            last_col = ori_col + step_col * grid_data.shape[1]

        else:
            raise RuntimeError(
                "Grid type {} not a path or CarsDataset".format(type(grid))
            )

        cols = np.arange(ori_col, last_col, step_col)
        rows = np.arange(ori_row, last_row, step_row)

        # create regular grid points positions
        points = (cols, rows)
        grid_row, grid_col = np.mgrid[
            ori_row:last_row:step_row, ori_col:last_col:step_col
        ]
        sensor_row_positions = row_dep + grid_row
        sensor_col_positions = col_dep + grid_col

        # interpolate sensor positions
        interp_row = interpolate.interpn(
            (cols, rows),
            sensor_row_positions.transpose(),
            positions,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        interp_col = interpolate.interpn(
            points,
            sensor_col_positions.transpose(),
            positions,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        # stack both coordinates
        sensor_positions = np.transpose(np.vstack([interp_col, interp_row]))
        return sensor_positions

    @staticmethod
    @abstractmethod
    def direct_loc(
        cars_conf,
        product_key: str,
        x_coord: float,
        y_coord: float,
        z_coord: float = None,
        dem: str = None,
        geoid: str = None,
        default_elevation: float = None,
    ) -> np.ndarray:
        """
        For a given image point, compute the latitude, longitude, altitude

        Advice: to be sure, use x,y,z inputs only

        :param cars_conf: cars input configuration dictionary
        :param product_key: input_parameters.PRODUCT1_KEY or
               input_parameters.PRODUCT2_KEY to identify which geometric model
               shall be taken to perform the method
        :param x_coord: X Coordinate in input image sensor
        :param y_coord: Y Coordinate in input image sensor
        :param z_coord: Z Altitude coordinate to take the image
        :param dem: if z not defined, take this DEM directory input
        :param geoid: if z and dem not defined, take GEOID directory input
        :param default_elevation: if z, dem, geoid not defined, take default
               elevation
        :return: Latitude, Longitude, Altitude coordinates as a numpy array
        """

    def image_envelope(
        self,
        conf,
        product_key: str,
        shp: str,
        dem: str = None,
        default_alt: float = None,
        geoid: str = None,
    ):
        """
        Export the image footprint to a shapefile

        :param conf: cars input configuration dictionary
        :param product_key: input_parameters.PRODUCT1_KEY or
               input_parameters.PRODUCT2_KEY to identify which geometric model
               shall be taken to perform the method
        :param shp: Path to the output shapefile
        :param dem: Directory containing DEM tiles
        :param default_alt: Default altitude above ellipsoid
        :param geoid: path to geoid file
        """
        # retrieve image size
        img = conf[
            input_parameters.create_img_tag_from_product_key(product_key)
        ]
        img_size_x, img_size_y = inputs.rasterio_get_size(img)

        # compute corners ground coordinates
        shift_x = -0.5
        shift_y = -0.5
        lat_upper_left, lon_upper_left, _ = self.direct_loc(
            conf,
            product_key,
            shift_x,
            shift_y,
            dem=dem,
            default_elevation=default_alt,
            geoid=geoid,
        )
        lat_upper_right, lon_upper_right, _ = self.direct_loc(
            conf,
            product_key,
            img_size_x + shift_x,
            shift_y,
            dem=dem,
            default_elevation=default_alt,
            geoid=geoid,
        )
        lat_bottom_left, lon_bottom_left, _ = self.direct_loc(
            conf,
            product_key,
            shift_x,
            img_size_y + shift_y,
            dem=dem,
            default_elevation=default_alt,
            geoid=geoid,
        )
        lat_bottom_right, lon_bottom_right, _ = self.direct_loc(
            conf,
            product_key,
            img_size_x + shift_x,
            img_size_y + shift_y,
            dem=dem,
            default_elevation=default_alt,
            geoid=geoid,
        )

        # create envelope polygon and save it as a shapefile
        poly_bb = Polygon(
            [
                (lon_upper_left, lat_upper_left),
                (lon_upper_right, lat_upper_right),
                (lon_bottom_right, lat_bottom_right),
                (lon_bottom_left, lat_bottom_left),
                (lon_upper_left, lat_upper_left),
            ]
        )

        outputs.write_vector([poly_bb], shp, 4326, driver="ESRI Shapefile")


def read_geoid_file(geoid_path: str) -> xr.Dataset:
    """
    Read geoid height from the given path
    Geoid is defined in the static configuration.

    Geoid is returned as an xarray.Dataset and height is stored in the `hgt`
    variable, which is indexed by `lat` and `lon` coordinates. Dataset
    attributes contain geoid bounds geodetic coordinates and
    latitude/longitude step spacing.

    :return: the geoid height array in meter.
    """
    with open(geoid_path, mode="rb") as in_grd:  # reading binary data
        # first header part, 4 float of 4 bytes -> 16 bytes to read
        # Endianness seems to be Big-Endian.
        lat_min, lat_max, lon_min, lon_max = struct.unpack(
            ">ffff", in_grd.read(16)
        )
        lat_step, lon_step = struct.unpack(">ff", in_grd.read(8))

        n_lats = int(np.ceil((lat_max - lat_min)) / lat_step) + 1
        n_lons = int(np.ceil((lon_max - lon_min)) / lon_step) + 1

        # read height grid.
        geoid_height = np.fromfile(in_grd, ">f4").reshape(n_lats, n_lons)

        # create output Dataset
        geoid = xr.Dataset(
            {"hgt": (("lat", "lon"), geoid_height)},
            coords={
                "lat": np.linspace(lat_max, lat_min, n_lats),
                "lon": np.linspace(lon_min, lon_max, n_lons),
            },
            attrs={
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "d_lat": lat_step,
                "d_lon": lon_step,
            },
        )

        return geoid
