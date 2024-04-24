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
from scipy.interpolate import LinearNDInterpolator
from shapely.geometry import Polygon

from cars.core import constants as cst
from cars.core import constants_disparity as cst_disp
from cars.core import inputs, outputs
from cars.data_structures import cars_dataset


class AbstractGeometry(metaclass=ABCMeta):
    """
    AbstractGeometry
    """

    available_plugins: Dict = {}

    def __new__(cls, geometry_plugin=None, **kwargs):
        """
        Return the required plugin
        :raises:
         - KeyError when the required plugin is not registered

        :param geometry_plugin: plugin name to instantiate
        :return: a geometry_plugin object
        """
        if geometry_plugin is not None:
            if geometry_plugin not in cls.available_plugins:
                logging.error(
                    "No geometry plugin named {} registered".format(
                        geometry_plugin
                    )
                )
                raise KeyError(
                    "No geometry plugin named {} registered".format(
                        geometry_plugin
                    )
                )

            logging.info(
                "The AbstractGeometry {} plugin will be used".format(
                    geometry_plugin
                )
            )

            return super(AbstractGeometry, cls).__new__(
                cls.available_plugins[geometry_plugin]
            )
        return super().__new__(cls)

    def __init__(
        self, geometry_plugin, dem=None, geoid=None, default_alt=None, **kwargs
    ):

        self.plugin_name = geometry_plugin
        self.dem = dem
        self.dem_roi = None
        self.dem_roi_epsg = None
        self.geoid = geoid
        self.default_alt = default_alt
        self.kwargs = kwargs

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
            cls.available_plugins[short_name] = subclass
            return subclass

        return decorator

    @staticmethod
    @abstractmethod
    def triangulate(
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        mode: str,
        matches: Union[xr.Dataset, np.ndarray],
        grid1: str,
        grid2: str,
        roi_key: Union[None, str] = None,
    ) -> np.ndarray:
        """
        Performs triangulation from cars disparity or matches dataset

        :param sensor1: path to left sensor image
        :param sensor2: path to right sensor image
        :param geomodel1: path and attributes for left geomodel
        :param geomodel2: path and attributes for right geomodel
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
    def check_product_consistency(sensor: str, geomodel: str, **kwargs) -> bool:
        """
        Test if the product is readable by the geometry plugin

        :param sensor: path to sensor image
        :param geomodel: path to geomodel
        :return: True if the products are readable, False otherwise
        """

    @abstractmethod
    def generate_epipolar_grids(
        self, sensor1, sensor2, geomodel1, geomodel2, epipolar_step: int = 30
    ) -> Tuple[
        np.ndarray, np.ndarray, List[float], List[float], List[int], float
    ]:
        """
        Computes the left and right epipolar grids

        :param sensor1: path to left sensor image
        :param sensor2: path to right sensor image
        :param geomodel1: path and attributes for left geomodel
        :param geomodel2: path and attributes for right geomodel
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

    def load_geomodel(self, geomodel: dict) -> dict:
        """
        By default return the geomodel
        This method can be overloaded by plugins to load geomodel in memory

        :param geomodel
        """
        return geomodel

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
        plugins if it requires matches in sensor coordinates to perform
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
            ori_col = grid.attributes["grid_origin"][1]
            ori_row = grid.attributes["grid_origin"][0]
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
    def epipolar_position_from_grid(grid, sensor_positions, step=30):
        """
        Compute epipolar position from grid

        :param grid: epipolar grid
        :param sensor_positions: sensor positions
        :param step: step of grid interpolator

        :return epipolar positions
        """
        # Generate interpolations grid to compute reverse

        epi_size_x = grid.attributes["epipolar_size_x"]
        epi_size_y = grid.attributes["epipolar_size_y"]

        epi_grid_row, epi_grid_col = np.mgrid[
            0:epi_size_x:step, 0:epi_size_y:step
        ]

        full_epi_pos = np.stack(
            [epi_grid_row.flatten(), epi_grid_col.flatten()], axis=1
        )

        sensor_interp_pos = AbstractGeometry.sensor_position_from_grid(
            grid, full_epi_pos
        )
        interp_row = LinearNDInterpolator(
            list(
                zip(  # noqa: B905
                    sensor_interp_pos[:, 0], sensor_interp_pos[:, 1]
                )
            ),
            epi_grid_row.flatten(),
        )
        epi_interp_row = interp_row(
            sensor_positions[:, 0], sensor_positions[:, 1]
        )

        interp_col = LinearNDInterpolator(
            list(
                zip(  # noqa: B905
                    sensor_interp_pos[:, 0], sensor_interp_pos[:, 1]
                )
            ),
            epi_grid_col.flatten(),
        )
        epi_interp_col = interp_col(
            sensor_positions[:, 0], sensor_positions[:, 1]
        )

        epipolar_positions = np.stack(
            (epi_interp_row, epi_interp_col)
        ).transpose()

        return epipolar_positions

    @staticmethod
    def transform_matches_from_grids(
        matches_array, grid_left, grid_right, new_grid_left, new_grid_right
    ):
        """
        Transform epipolar matches with grid transformation

        :param grid_left: path to epipolar grid of image 1
        :param grid_left: path to epipolar grid of image 2
        :param new_grid_left: path to epipolar grid of image 1
        :param new_grid_right: path to epipolar grid of image 2
        :param matches: cars disparity dataset or matches as numpy array

        """

        # Transform to sensors
        sensor_matches_left = AbstractGeometry.sensor_position_from_grid(
            grid_left, matches_array[:, 0:2]
        )
        sensor_matches_right = AbstractGeometry.sensor_position_from_grid(
            grid_right, matches_array[:, 2:4]
        )

        # Transform to new grids
        new_grid_matches_left = AbstractGeometry.epipolar_position_from_grid(
            new_grid_left, sensor_matches_left
        )
        new_grid_matches_right = AbstractGeometry.epipolar_position_from_grid(
            new_grid_right, sensor_matches_right
        )

        # Concatenate matches
        new_matches_array = np.concatenate(
            [new_grid_matches_left, new_grid_matches_right], axis=1
        )

        # Linear interpolation might generate nan on the borders
        new_matches_array = new_matches_array[
            ~np.isnan(new_matches_array).any(axis=1)
        ]

        return new_matches_array

    @abstractmethod
    def direct_loc(
        self,
        sensor,
        geomodel,
        x_coord: np.array,
        y_coord: np.array,
        z_coord: np.array = None,
    ) -> np.ndarray:
        """
        For a given image points list, compute the latitudes,
        longitudes, altitudes

        Advice: to be sure, use x,y,z list inputs only

        :param sensor: path to sensor image
        :param geomodel: path and attributes for geomodel
        :param x_coord: X Coordinates list in input image sensor
        :param y_coord: Y Coordinate list in input image sensor
        :param z_coord: Z Altitude list coordinate to take the image
        :return: Latitude, Longitude, Altitude coordinates list as a numpy array
        """

    @abstractmethod
    def inverse_loc(
        self,
        sensor,
        geomodel,
        lat_coord: np.array,
        lon_coord: np.array,
        z_coord: np.array = None,
    ) -> np.ndarray:
        """
        For a given image points list, compute the latitudes,
        longitudes, altitudes

        Advice: to be sure, use x,y,z list inputs only

        :param sensor: path to sensor image
        :param geomodel: path and attributes for geomodel
        :param lat_coord: latitute Coordinate list
        :param lon_coord: longitude Coordinates list
        :param z_coord: Z Altitude list
        :return: X  / Y / Z Coordinates list in input image as a numpy array
        """

    def sensors_arrangement_left_right(
        self, sensor1, sensor2, geomodel1, geomodel2, grid_left, grid_right
    ):
        """
        Determine the arrangement of sensors, either:
        (double slashes represent Lines Of Sight)
        +---------------------+---------------------+
        |    Arrangement 1    |    Arrangement 2    |
        |  sensor1   sensor2  |   sensor2  sensor1  |
        |   \\ \\    //       |       \\    // //   |
        |    \\ \\  //        |        \\  // //    |
        |     \\ \\// <-- z_2 | z_1 --> \\// //     |
        |      \\ //          |          \\ //      |
        |       \\/   <-- z_1 | z_2 -->   \\/       |
        +---------------------+---------------------+
        This allows to know if a lower disparity corresponds to a lower or
        higher hgt. It depends on the image pairing and geometrical models.
        A fake triangulation determines z_1 and z_2.
        If z_1 < z_2 then the sensors are in arrangement 1
        If z_2 < z_1 then the sensors are in arrangement 2

        :param sensor1: path to left sensor image
        :param sensor2: path to right sensor image
        :param geomodel1: path and attributes for left geomodel
        :param geomodel2: path and attributes for right geomodel
        :param grid1: path or dataset for epipolar grid of sensor1
        :param grid2: path or dataset for epipolar grid of sensor2

        :return: boolean indicating if sensors are in arrangement 1 or not
        """
        # Create a fake disparity dataset, where the two LOS from
        # sensor1 are associated with the same LOS from sensor2
        fake_disp = xr.Dataset(
            data_vars={
                cst_disp.MAP: (
                    [cst.ROW, cst.COL],
                    np.array([[1, 0]], dtype=float),
                )
            },
            coords={cst.ROW: [0], cst.COL: [0, 1]},
            attrs={cst.ROI: [0, 0, 2, 1], cst.EPI_FULL_SIZE: [2, 1]},
        )
        fake_triangulation = self.triangulate(
            sensor1,
            sensor2,
            geomodel1,
            geomodel2,
            cst.DISP_MODE,
            fake_disp,
            grid_left,
            grid_right,
            roi_key=cst.ROI,
        )
        # True if arrangement 1, False if arrangement 2
        return fake_triangulation[0, 0, 2] < fake_triangulation[0, 1, 2]

    def image_envelope(self, sensor, geomodel, shp=None):
        """
        Export the image footprint to a shapefile

        :param sensor: path to sensor image
        :param geomodel: path and attributes for geometrical model
        :param shp: Path to the output shapefile
        """
        # retrieve image size
        img_size_x, img_size_y = inputs.rasterio_get_size(sensor)

        # compute corners ground coordinates
        shift_x = -0.5
        shift_y = -0.5
        # TODO call 1 time with multipoint
        lat_upper_left, lon_upper_left, _ = self.direct_loc(
            sensor,
            geomodel,
            np.array(shift_x),
            np.array(shift_y),
        )
        lat_upper_right, lon_upper_right, _ = self.direct_loc(
            sensor,
            geomodel,
            np.array(img_size_x + shift_x),
            np.array(shift_y),
        )
        lat_bottom_left, lon_bottom_left, _ = self.direct_loc(
            sensor,
            geomodel,
            np.array(shift_x),
            np.array(img_size_y + shift_y),
        )
        lat_bottom_right, lon_bottom_right, _ = self.direct_loc(
            sensor,
            geomodel,
            np.array(img_size_x + shift_x),
            np.array(img_size_y + shift_y),
        )

        u_l = (lon_upper_left, lat_upper_left)
        u_r = (lon_upper_right, lat_upper_right)
        l_l = (lon_bottom_left, lat_bottom_left)
        l_r = (lon_bottom_right, lat_bottom_right)

        if shp is not None:
            # create envelope polygon and save it as a shapefile
            poly_bb = Polygon([u_l, u_r, l_r, l_l, u_l])
            outputs.write_vector([poly_bb], shp, 4326, driver="ESRI Shapefile")

        return u_l, u_r, l_l, l_r


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
