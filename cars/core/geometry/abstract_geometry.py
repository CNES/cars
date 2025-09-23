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

# pylint: disable=C0302

"""
this module contains the abstract geometry class to use in the
geometry plugins
"""
import logging
import os
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import rasterio as rio
import xarray as xr
from json_checker import And, Checker, Or
from scipy import interpolate
from scipy.interpolate import LinearNDInterpolator
from shapely.geometry import Polygon
from shareloc import proj_utils
from shareloc.geofunctions.rectification_grid import RectificationGrid

from cars.core import constants as cst
from cars.core import inputs, outputs
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset
from cars.orchestrator.cluster.log_wrapper import cars_profile


class AbstractGeometry(metaclass=ABCMeta):
    """
    AbstractGeometry
    """

    available_plugins: Dict = {}

    def __new__(cls, geometry_plugin_conf=None, scaling_coeff=1, **kwargs):
        """
        Return the required plugin
        :raises:
         - KeyError when the required plugin is not registered

        :param geometry_plugin_conf: plugin name or plugin configuration
            to instantiate
        :type geometry_plugin_conf: str or dict
        :param scaling_coeff: scaling factor for resolution
        :type scaling_coeff: float
        :return: a geometry_plugin object
        """
        if geometry_plugin_conf is not None:
            if isinstance(geometry_plugin_conf, str):
                geometry_plugin = geometry_plugin_conf
            elif isinstance(geometry_plugin_conf, dict):
                geometry_plugin = geometry_plugin_conf.get(
                    "plugin_name", "SharelocGeometry"
                )
            else:
                raise RuntimeError("Not a supported type")

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
        self,
        geometry_plugin_conf,
        dem=None,
        geoid=None,
        default_alt=None,
        scaling_coeff=1,
        **kwargs,
    ):

        self.scaling_coeff = scaling_coeff

        config = self.check_conf(geometry_plugin_conf)

        self.plugin_name = config["plugin_name"]
        self.interpolator = config["interpolator"]
        self.dem_roi_margin = config["dem_roi_margin"]

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

    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check
        :type conf: str or dict

        :return: full dict
        :rtype: dict

        """

        if conf is None:
            raise RuntimeError("Geometry plugin configuration is None")

        overloaded_conf = {}

        if isinstance(conf, str):
            conf = {"plugin_name": conf}

        # overload conf
        overloaded_conf["plugin_name"] = conf.get(
            "plugin_name", "SharelocGeometry"
        )
        overloaded_conf["interpolator"] = conf.get("interpolator", "cubic")
        overloaded_conf["dem_roi_margin"] = conf.get("dem_roi_margin", 0.012)

        geometry_schema = {
            "plugin_name": str,
            "interpolator": And(str, lambda x: x in ["cubic", "linear"]),
            "dem_roi_margin": Or(float, int),
        }

        # Check conf
        checker = Checker(geometry_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    @abstractmethod
    def triangulate(
        self,
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

    def matches_to_sensor_coords(
        self,
        grid1: Union[str, cars_dataset.CarsDataset, RectificationGrid],
        grid2: Union[str, cars_dataset.CarsDataset, RectificationGrid],
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
        sensor_pos_left = self.sensor_position_from_grid(
            grid1, vec_epi_pos_left
        )
        sensor_pos_right = self.sensor_position_from_grid(
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

    def sensor_position_from_grid(
        self,
        grid: Union[dict, RectificationGrid],
        positions: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate the positions given as inputs using the grid

        :param grid: rectification grid dict, or RectificationGrid object
        :type grid: Union[dict, RectificationGrid]
        :param positions: epipolar positions to interpolate given as a numpy
               array of size [number of points, 2]. The last index indicates
               the 'x' coordinate (last index set to 0) or the 'y' coordinate
               (last index set to 1).
        :return: sensors positions as a numpy array of size
                 [number of points, 2]. The last index indicates the 'x'
                 coordinate (last index set to 0) or
                 the 'y' coordinate (last index set to 1).
        """

        if isinstance(grid, RectificationGrid):
            return grid.interpolate(positions)

        if not isinstance(grid, dict):
            raise RuntimeError(
                f"Grid type {type(grid)} not a dict or RectificationGrid"
            )

        # Ensure positions is a numpy array
        positions = np.asarray(positions)

        # Get data
        with rio.open(grid["path"]) as grid_data:
            row_dep = grid_data.read(2)
            col_dep = grid_data.read(1)

        # Get step
        step_col = grid["grid_spacing"][1]
        step_row = grid["grid_spacing"][0]
        ori_col = grid["grid_origin"][1]
        ori_row = grid["grid_origin"][0]
        last_col = ori_col + step_col * row_dep.shape[1]
        last_row = ori_row + step_row * row_dep.shape[0]

        cols = np.arange(ori_col, last_col, step_col)
        rows = np.arange(ori_row, last_row, step_row)

        # Determine margin based on interpolator type
        margin = 6 if self.interpolator == "cubic" else 3

        # Find the bounds of positions to determine crop region
        min_col = np.nanmin(positions[:, 0])
        max_col = np.nanmax(positions[:, 0])
        min_row = np.nanmin(positions[:, 1])
        max_row = np.nanmax(positions[:, 1])

        # Convert position bounds to grid indices with margin
        min_col_idx = max(0, int((min_col - ori_col) / step_col) - margin)
        max_col_idx = min(
            len(cols) - 1, int((max_col - ori_col) / step_col) + margin
        )
        min_row_idx = max(0, int((min_row - ori_row) / step_row) - margin)
        max_row_idx = min(
            len(rows) - 1, int((max_row - ori_row) / step_row) + margin
        )

        # Crop the grids and coordinate arrays
        cols_cropped = cols[min_col_idx : max_col_idx + 1]
        rows_cropped = rows[min_row_idx : max_row_idx + 1]
        sensor_row_positions_cropped = row_dep[
            min_row_idx : max_row_idx + 1, min_col_idx : max_col_idx + 1
        ]
        sensor_col_positions_cropped = col_dep[
            min_row_idx : max_row_idx + 1, min_col_idx : max_col_idx + 1
        ]

        # interpolate sensor positions
        interpolator = interpolate.RegularGridInterpolator(
            (cols_cropped, rows_cropped),
            np.stack(
                (
                    sensor_row_positions_cropped.transpose(),
                    sensor_col_positions_cropped.transpose(),
                ),
                axis=2,
            ),
            method=self.interpolator,
            bounds_error=False,
            fill_value=None,
        )

        sensor_positions = interpolator(positions)

        # swap
        sensor_positions[:, [0, 1]] = sensor_positions[:, [1, 0]]

        return sensor_positions

    def epipolar_position_from_grid(self, grid, sensor_positions, step=30):
        """
        Compute epipolar position from grid

        :param grid: epipolar grid
        :param sensor_positions: sensor positions
        :param step: step of grid interpolator

        :return epipolar positions
        """
        # Generate interpolations grid to compute reverse

        epi_size_x = grid["epipolar_size_x"]
        epi_size_y = grid["epipolar_size_y"]

        epi_grid_row, epi_grid_col = np.mgrid[
            0:epi_size_x:step, 0:epi_size_y:step
        ]

        full_epi_pos = np.stack(
            [epi_grid_row.flatten(), epi_grid_col.flatten()], axis=1
        )

        sensor_interp_pos = self.sensor_position_from_grid(grid, full_epi_pos)
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

    @cars_profile(name="Transform matches", interval=0.5)
    def transform_matches_from_grids(
        self,
        sensor_matches_left,
        sensor_matches_right,
        new_grid_left,
        new_grid_right,
    ):
        """
        Transform epipolar matches with grid transformation

        :param new_grid_left: path to epipolar grid of image 1
        :param new_grid_right: path to epipolar grid of image 2
        :param matches: cars disparity dataset or matches as numpy array

        """

        # Transform to new grids
        new_grid_matches_left = self.epipolar_position_from_grid(
            new_grid_left, sensor_matches_left
        )
        new_grid_matches_right = self.epipolar_position_from_grid(
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

    @cars_profile(name="Get sensor matches")
    def get_sensor_matches(
        self,
        matches_array,
        grid_left,
        grid_right,
        pair_folder,
        save_matches,
    ):
        """
        get sensor matches

        :param grid_left: path to epipolar grid of image 1
        :param grid_left: path to epipolar grid of image 2
        """
        # Transform to sensors
        sensor_matches_left = self.sensor_position_from_grid(
            grid_left, matches_array[:, 0:2]
        )
        sensor_matches_right = self.sensor_position_from_grid(
            grid_right, matches_array[:, 2:4]
        )

        current_out_dir = None
        if save_matches:
            logging.info("Writing matches file")
            if pair_folder is None:
                logging.error("Pair folder not provided")
            else:
                safe_makedirs(pair_folder)
                current_out_dir = pair_folder
            matches_sensor_left_path = os.path.join(
                current_out_dir, "sensor_matches_left.npy"
            )
            matches_sensor_right_path = os.path.join(
                current_out_dir, "sensor_matches_right.npy"
            )
            np.save(matches_sensor_left_path, sensor_matches_left)
            np.save(matches_sensor_right_path, sensor_matches_right)

        return sensor_matches_left, sensor_matches_right

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

    def safe_direct_loc(
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
        if len(x_coord) > 0:
            ground_points = self.direct_loc(
                sensor,
                geomodel,
                x_coord,
                y_coord,
                z_coord,
            )
        else:
            logging.warning("Direct loc function launched on empty list")
            return []
        if z_coord is None:
            status = np.any(np.isnan(ground_points), axis=0)
            if sum(status) > 0:
                logging.warning(
                    "{} errors have been detected on direct "
                    "loc and will be re-launched".format(sum(status))
                )
                ground_points_retry = self.direct_loc(
                    sensor,
                    geomodel,
                    x_coord[status],
                    y_coord[status],
                    np.array([0]),
                )
                ground_points[:, status] = ground_points_retry
        return ground_points

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

    def safe_inverse_loc(
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
        if len(lat_coord) > 0:
            image_points = self.inverse_loc(
                sensor,
                geomodel,
                lat_coord,
                lon_coord,
                z_coord,
            )
            image_points = np.array(image_points)
        else:
            logging.warning("Inverse loc function launched on empty list")
            return [], [], []
        if z_coord is None:
            image_points = np.array(image_points)
            status = np.any(np.isnan(image_points), axis=0)
            if sum(status) > 0:
                logging.warning(
                    "{} errors have been detected on inverse "
                    "loc and will be re-launched".format(sum(status))
                )
            image_points_retry = self.inverse_loc(
                sensor,
                geomodel,
                lat_coord[status],
                lon_coord[status],
                np.array([self.default_alt]),
            )

            image_points[:, status] = image_points_retry
        return image_points[0], image_points[1], image_points[2]

    def image_envelope(
        self,
        sensor,
        geomodel,
        out_path=None,
        out_driver="ESRI Shapefile",
        elevation=None,
    ):
        """
        Export the image footprint to a vector file

        :param sensor: path to sensor image
        :param geomodel: path and attributes for geometrical model
        :param out_path: Path to the output vector file
        :param out_driver: OGR driver to use to write output file
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
            elevation,
        )
        lat_upper_right, lon_upper_right, _ = self.direct_loc(
            sensor,
            geomodel,
            np.array(img_size_x + shift_x),
            np.array(shift_y),
            elevation,
        )
        lat_bottom_left, lon_bottom_left, _ = self.direct_loc(
            sensor,
            geomodel,
            np.array(shift_x),
            np.array(img_size_y + shift_y),
            elevation,
        )
        lat_bottom_right, lon_bottom_right, _ = self.direct_loc(
            sensor,
            geomodel,
            np.array(img_size_x + shift_x),
            np.array(img_size_y + shift_y),
            elevation,
        )

        u_l = (lon_upper_left, lat_upper_left)
        u_r = (lon_upper_right, lat_upper_right)
        l_l = (lon_bottom_left, lat_bottom_left)
        l_r = (lon_bottom_right, lat_bottom_right)

        if out_path is not None:
            # create envelope polygon and save it as a shapefile
            poly_bb = Polygon([u_l, u_r, l_r, l_l, u_l])
            outputs.write_vector([poly_bb], out_path, 4326, driver=out_driver)

        return u_l, u_r, l_l, l_r


def min_max_to_physical_min_max(xmin, xmax, ymin, ymax, transform):
    """
    Transform min max index to position min max

    :param xmin: xmin
    :type xmin: int
    :param xmax: xmax
    :type xmax: int
    :param ymin: ymin
    :type ymin: int
    :param ymax: ymax
    :type ymax: int
    :param transform: transform
    :type transform: Affine

    :return: xmin, xmax, ymin, ymax
    :rtype: list(int)
    """

    cols_ind = np.array([xmin, xmin, xmax, xmax])
    rows_ind = np.array([ymin, ymax, ymin, ymax])

    rows_pos, cols_pos = proj_utils.transform_index_to_physical_point(
        transform,
        rows_ind,
        cols_ind,
    )

    return (
        np.min(cols_pos),
        np.max(cols_pos),
        np.min(rows_pos),
        np.max(rows_pos),
    )


def min_max_to_index_min_max(xmin, xmax, ymin, ymax, transform):
    """
    Transform min max position to index min max

    :param xmin: xmin
    :type xmin: int
    :param xmax: xmax
    :type xmax: int
    :param ymin: ymin
    :type ymin: int
    :param ymax: ymax
    :type ymax: int
    :param transform: transform
    :type transform: Affine

    :return: xmin, xmax, ymin, ymax
    :rtype: list(int)
    """

    cols_ind = np.array([xmin, xmin, xmax, xmax])
    rows_ind = np.array([ymin, ymax, ymin, ymax])

    rows_pos, cols_pos = proj_utils.transform_physical_point_to_index(
        ~transform,
        rows_ind,
        cols_ind,
    )

    return (
        np.min(cols_pos),
        np.max(cols_pos),
        np.min(rows_pos),
        np.max(rows_pos),
    )
