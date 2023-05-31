#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2023 Centre National d'Etudes Spatiales (CNES).
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
Shareloc geometry sub class : CARS geometry wrappers functions to shareloc ones
"""

import logging
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import rasterio as rio
import shareloc.geofunctions.rectification as rectif
import xarray as xr
from json_checker import And
from shareloc.geofunctions import localization
from shareloc.geofunctions.dtm_intersection import DTMIntersection
from shareloc.geofunctions.triangulation import epipolar_triangulation
from shareloc.geomodels.grid import Grid
from shareloc.geomodels.rpc import RPC
from shareloc.image import Image

from cars.conf import input_parameters
from cars.core import constants as cst
from cars.core.geometry import AbstractGeometry

GRID_TYPE = "GRID"
RPC_TYPE = "RPC"


@AbstractGeometry.register_subclass("SharelocGeometry")
class SharelocGeometry(AbstractGeometry):
    """
    shareloc geometry class
    """

    @property
    def conf_schema(self) -> Dict[str, str]:
        """
        Defines shareloc loader user configuration specificities

        TODO: not needed
        To remove, with abstract geometry class evolution
        and with CARS former configuration update (with OTB evolution)
        """

        schema = {
            input_parameters.IMG1_TAG: And(str, os.path.isfile),
            input_parameters.IMG2_TAG: And(str, os.path.isfile),
            input_parameters.MODEL1_TAG: And(str, os.path.isfile),
            input_parameters.MODEL2_TAG: And(str, os.path.isfile),
            input_parameters.MODEL1_TYPE_TAG: str,
            input_parameters.MODEL2_TYPE_TAG: str,
        }

        return schema

    @staticmethod
    def load_geom_model(model: str, model_type: str) -> Union[Grid, RPC]:
        """
        Load geometric model and returns it as a shareloc object

        TODO: evolve with CARS new API with CARS conf clean

        :param model: Path to the model file
        :param model_type: model type (RPC or Grid)
        :return: geometric model as a shareloc object (Grid or RPC)
        """

        if model_type == GRID_TYPE:
            shareloc_model = Grid(model)
        elif model_type == RPC_TYPE:
            shareloc_model = RPC.from_any(model)
        else:
            raise ValueError(f"Model type {model_type} is not supported")

        if shareloc_model is None:
            raise ValueError(f"Model {model} could not be read by shareloc")

        return shareloc_model

    @staticmethod
    def load_image(img: str) -> Image:
        """
        Load the image using the Image class of Shareloc

        :param img: path to the image
        :return: The Image object
        """
        try:
            shareloc_img = Image(img)
        except Exception as error:
            raise ValueError(f"Image type {img} is not supported") from error

        return shareloc_img

    @staticmethod
    def check_products_consistency(cars_conf) -> bool:
        """
        Test if the product is readable by the shareloc loader

        TODO: not used
        - to evolve and use in CARS configuration early in pipeline process
        (new early check input common application ?)
        - update with former cars clean with otb evolution.

        :param cars_conf: cars input configuration dictionary
        :return: True if the products are readable, False otherwise
        """
        # get inputs paths
        img1 = cars_conf[input_parameters.IMG1_TAG]
        img2 = cars_conf[input_parameters.IMG2_TAG]
        model1 = cars_conf[input_parameters.MODEL1_TAG]
        model2 = cars_conf[input_parameters.MODEL2_TAG]
        model1_type = cars_conf[input_parameters.MODEL1_TYPE_TAG]
        model2_type = cars_conf[input_parameters.MODEL2_TYPE_TAG]

        # Try to read them using shareloc
        status = True
        try:
            SharelocGeometry.load_image(img1)
            SharelocGeometry.load_image(img2)
            SharelocGeometry.load_geom_model(model1, model1_type)
            SharelocGeometry.load_geom_model(model2, model2_type)
        except Exception:
            status = False

        return status

    @staticmethod
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

        TODO: evolve with CARS new API with CARS conf clean

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
        # get inputs paths
        model1 = cars_conf[input_parameters.MODEL1_TAG]
        model2 = cars_conf[input_parameters.MODEL2_TAG]
        model1_type = cars_conf[input_parameters.MODEL1_TYPE_TAG]
        model2_type = cars_conf[input_parameters.MODEL2_TYPE_TAG]

        # read them using shareloc
        shareloc_model1 = SharelocGeometry.load_geom_model(model1, model1_type)
        shareloc_model2 = SharelocGeometry.load_geom_model(model2, model2_type)

        # perform matches triangulation
        if mode is cst.MATCHES_MODE:
            __, point_wgs84, __ = epipolar_triangulation(
                matches=matches,
                mask=None,
                matches_type="sift",
                geometrical_model_left=shareloc_model1,
                geometrical_model_right=shareloc_model2,
                grid_left=grid1,
                grid_right=grid2,
                residues=True,
                fill_nan=True,
            )

            llh = point_wgs84.reshape((point_wgs84.shape[0], 1, 3))

        elif mode is cst.DISP_MODE:
            __, point_wgs84, __ = epipolar_triangulation(
                matches=matches,
                mask=None,
                matches_type="disp",
                geometrical_model_left=shareloc_model1,
                geometrical_model_right=shareloc_model2,
                grid_left=grid1,
                grid_right=grid2,
                residues=True,
                fill_nan=True,
            )

            row = np.array(
                range(matches.attrs[roi_key][1], matches.attrs[roi_key][3])
            )
            col = np.array(
                range(matches.attrs[roi_key][0], matches.attrs[roi_key][2])
            )

            llh = point_wgs84.reshape((row.size, col.size, 3))
        else:
            logging.error(
                "{} mode is not available in the "
                "shareloc loader triangulation".format(mode)
            )
            raise ValueError(
                f"{mode} mode is not available"
                " in the shareloc loader triangulation"
            )

        return llh

    @staticmethod
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

        TODO: evolve with CARS new API with CARS conf clean

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
            - the epipolar image size as a list of int
            (x-axis size is given with the index 0, y-axis size with index 1)
            - the disparity to altitude ratio as a float

        """
        # get inputs paths and value
        model1 = cars_conf[input_parameters.MODEL1_TAG]
        model2 = cars_conf[input_parameters.MODEL2_TAG]
        img1 = cars_conf[input_parameters.IMG1_TAG]
        img2 = cars_conf[input_parameters.IMG2_TAG]
        model1_type = cars_conf[input_parameters.MODEL1_TYPE_TAG]
        model2_type = cars_conf[input_parameters.MODEL2_TYPE_TAG]

        # read inputs using shareloc
        shareloc_model1 = SharelocGeometry.load_geom_model(model1, model1_type)
        shareloc_model2 = SharelocGeometry.load_geom_model(model2, model2_type)

        image1 = Image(img1)
        image2 = Image(img2)

        # set elevation from geoid/dem/default_alt
        if dem is not None:
            extent = rectif.get_epipolar_extent(
                image1, shareloc_model1, shareloc_model2, margin=0.0056667
            )
            # fill_nodata option should be set when dealing with void in DTM
            # see shareloc DTM limitations in sphinx doc for further details
            elevation = DTMIntersection(
                dem, geoid, roi=extent, fill_nodata="mean"
            )
        else:
            elevation = default_alt

        # compute epipolar grids
        (
            grid1,
            grid2,
            epipolar_size_y,
            epipolar_size_x,
            alt_to_disp_ratio,
        ) = rectif.compute_stereorectification_epipolar_grids(
            image1,
            shareloc_model1,
            image2,
            shareloc_model2,
            elevation,
            epi_step=epipolar_step,
        )

        # rearrange output to match the expected structure of CARS
        grid1 = np.moveaxis(grid1.data[::-1, :, :], 0, -1)
        grid2 = np.moveaxis(grid2.data[::-1, :, :], 0, -1)

        # compute associated characteristics
        with rio.open(img1, "r") as rio_dst:
            pixel_size_x, pixel_size_y = (
                rio_dst.transform[0],
                rio_dst.transform[4],
            )

        mean_size = (abs(pixel_size_x) + abs(pixel_size_y)) / 2
        epipolar_size_x = int(np.floor(epipolar_size_x * mean_size))
        epipolar_size_y = int(np.floor(epipolar_size_y * mean_size))

        origin = [0.0, 0.0]
        spacing = [float(epipolar_step), float(epipolar_step)]

        disp_to_alt_ratio = 1 / alt_to_disp_ratio

        return (
            grid1,
            grid2,
            origin,
            spacing,
            [epipolar_size_x, epipolar_size_y],
            disp_to_alt_ratio,
        )

    @staticmethod
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

        TODO: evolve with CARS new API with CARS conf clean

        :param cars_conf: cars input configuration dictionary
        :param product_key: input_parameters.PRODUCT1_KEY or
         input_parameters.PRODUCT2_KEY to identify which geometric model shall
         be taken to perform the method
        :param x_coord: X Coordinate in input image sensor
        :param y_coord: Y Coordinate in input image sensor
        :param z_coord: Z Altitude coordinate to take the image
        :param dem: if z not defined, take this DEM directory input
        :param geoid: if z and dem not defined, take GEOID directory input
        :param default_elevation: if z, dem, geoid not defined, take default
          elevation
        :return: Latitude, Longitude, Altitude coordinates as a numpy array
        """
        # read required product paths and model type
        model = cars_conf[
            input_parameters.create_model_tag_from_product_key(product_key)
        ]
        image = cars_conf[
            input_parameters.create_img_tag_from_product_key(product_key)
        ]
        model_type = cars_conf[
            input_parameters.create_model_type_tag_from_product_key(product_key)
        ]

        # load model and image with shareloc
        shareloc_model = SharelocGeometry.load_geom_model(model, model_type)
        shareloc_image = Image(image)

        # set elevation from geoid/dem/default_alt
        if dem is not None:
            # fill_nodata option should be set when dealing with void in DTM
            # see shareloc DTM limitations in sphinx doc for further details
            elevation = DTMIntersection(dem, geoid, fill_nodata="mean")
        else:
            elevation = default_elevation

        # perform direct localization operation
        loc = localization.Localization(
            shareloc_model, image=shareloc_image, elevation=elevation, epsg=4326
        )
        # Bug: y_coord and x_coord inversion to fit Shareloc standards row/col.
        # TODO: clean geometry convention calls in API
        lonlatalt = loc.direct(
            y_coord, x_coord, z_coord, using_geotransform=True
        )
        lonlatalt = np.squeeze(lonlatalt)
        latlonalt = np.array([lonlatalt[1], lonlatalt[0], lonlatalt[2]])
        return latlonalt
