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
this module contains the otb geometry class
"""
import logging
import os
from typing import List, Tuple, Union

import numpy as np
import otbApplication  # pylint: disable=import-error
import rasterio as rio
import xarray as xr
from json_checker import And

from cars.conf import input_parameters
from cars.core import constants as cst
from cars.core import constants_disparity as cst_disp
from cars.core import inputs
from cars.core.geometry import AbstractGeometry
from cars.core.otb_adapters import encode_to_otb
from cars.core.utils import get_elevation_range_from_metadata


@AbstractGeometry.register_subclass("OTBGeometry")
class OTBGeometry(AbstractGeometry):
    """
    OTB geometry class
    """

    # TODO: remove the hard-coded import in the steps/__init__.py if this class
    # is removed from CARS

    @property
    def conf_schema(self):
        """
        Returns the input configuration fields required by the geometry loader
        as a json checker schema. The available fields are defined in the
        cars/conf/input_parameters.py file

        :return: the geo configuration schema
        """

        geo_conf_schema = {
            input_parameters.IMG1_TAG: And(str, inputs.rasterio_can_open),
            input_parameters.IMG2_TAG: And(str, inputs.rasterio_can_open),
        }

        return geo_conf_schema

    @staticmethod
    def check_otb_remote_modules() -> List[str]:
        """
        Check all remote module compiled by cars
        :return list of not available modules
        """
        not_available = []
        for module in ["ConvertSensorToGeoPointFast", "EpipolarTriangulation"]:
            if otbApplication.Registry.CreateApplication(module) is None:
                not_available.append(module)
        return not_available

    @staticmethod
    def check_products_consistency(cars_conf) -> bool:
        """
        Test if the product is readable by the OTB

        :param: cars_conf: cars input configuration dictionary
        :return: True if the products are readable, False otherwise
        """
        # get images paths
        img1 = cars_conf[input_parameters.IMG1_TAG]
        img2 = cars_conf[input_parameters.IMG2_TAG]

        # test if both images have associated RPC
        status = False
        if OTBGeometry.check_geom_consistency(
            img1
        ) and OTBGeometry.check_geom_consistency(img2):
            status = True

        return status

    @staticmethod
    def check_geom_consistency(img: str) -> bool:
        """
        Check if the image have RPC information readable by the OTB

        :param img: path to the image
        :return: True if the RPC are readable, False otherwise
        """
        can_open_status = False
        try:
            geom_path = "./otb_can_open_test.geom"

            # try to dump .geom with ReadImageInfo app
            read_im_app = otbApplication.Registry.CreateApplication(
                "ReadImageInfo"
            )
            read_im_app.SetParameterString("in", img)
            read_im_app.SetParameterString("outkwl", geom_path)

            read_im_app.ExecuteAndWriteOutput()

            # check geom consistency
            if os.path.exists(geom_path):
                can_open_status = True

                with open(geom_path, encoding="utf-8") as geom_file_desc:
                    geom_dict = {}
                    for line in geom_file_desc:
                        key, val = line.split(": ")
                        geom_dict[key] = val
                    # pylint: disable=too-many-boolean-expressions
                    if (
                        "line_den_coeff_00" not in geom_dict
                        or "samp_den_coeff_00" not in geom_dict
                        or "line_num_coeff_00" not in geom_dict
                        or "samp_num_coeff_00" not in geom_dict
                        or "line_off" not in geom_dict
                        or "line_scale" not in geom_dict
                        or "samp_off" not in geom_dict
                        or "samp_scale" not in geom_dict
                        or "lat_off" not in geom_dict
                        or "lat_scale" not in geom_dict
                        or "long_off" not in geom_dict
                        or "long_scale" not in geom_dict
                        or "height_off" not in geom_dict
                        or "height_scale" not in geom_dict
                        or "polynomial_format" not in geom_dict
                    ):
                        logging.warning(
                            "No RPC model set for image {}".format(
                                geom_file_desc
                            )
                        )
                        can_open_status = False

                os.remove("./otb_can_open_test.geom")
            else:
                logging.warning(
                    "{} does not have associated geom file".format(img)
                )
                can_open_status = False
        except Exception as read_error:
            logging.warning(
                "Exception caught while trying to read file {}: {}".format(
                    img, read_error
                )
            )
            can_open_status = False
        return can_open_status

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

        :param cars_conf: cars input configuration dictionary
        :param mode: triangulation mode
               (cst.DISP_MODE or cst.MATCHES)
        :param matches: cars disparity dataset or matches as numpy array
        :param grid1: path to epipolar grid of image 1
        :param grid2: path to epipolar grid of image 2
        :param roi_key: dataset roi to use
               (can be cst.ROI or cst.ROI_WITH_MARGINS)
        :return: the long/lat/height numpy array in output of the triangulation
        """

        img1 = cars_conf[input_parameters.IMG1_TAG]
        img2 = cars_conf[input_parameters.IMG2_TAG]

        # Retrieve elevation range from imgs
        (min_elev1, max_elev1) = get_elevation_range_from_metadata(img1)
        (min_elev2, max_elev2) = get_elevation_range_from_metadata(img2)

        # Build triangulation app
        triangulation_app = otbApplication.Registry.CreateApplication(
            "EpipolarTriangulation"
        )

        if mode == cst.DISP_MODE:
            if roi_key is None:
                logging.warning(
                    "roi_key have to be set to use the "
                    "triangulation disparity mode"
                )
                raise RuntimeError(
                    "roi_key have to be set to use the "
                    "triangulation disparity mode"
                )

            # encode disparity for otb
            disp = encode_to_otb(
                matches[cst_disp.MAP].values,
                matches.attrs[cst.EPI_FULL_SIZE],
                matches.attrs[roi_key],
            )

            # set disparity mode
            triangulation_app.SetParameterString("mode", "disp")
            triangulation_app.ImportImage("mode.disp.indisp", disp)
        elif mode == cst.MATCHES_MODE:
            # set matches mode
            triangulation_app.SetParameterString("mode", "sift")
            triangulation_app.SetImageFromNumpyArray(
                "mode.sift.inmatches", matches
            )
        else:
            logging.warning(
                "Wrong triangulation mode (only 'disp' or 'matches'"
                " can be used as mode)"
            )
            raise NameError(
                "Wrong triangulation mode (only 'disp' or 'matches'"
                " can be used as mode)"
            )

        triangulation_app.SetParameterString("leftgrid", grid1)
        triangulation_app.SetParameterString("rightgrid", grid2)
        triangulation_app.SetParameterString("leftimage", img1)
        triangulation_app.SetParameterString("rightimage", img2)
        triangulation_app.SetParameterFloat("leftminelev", min_elev1)
        triangulation_app.SetParameterFloat("leftmaxelev", max_elev1)
        triangulation_app.SetParameterFloat("rightminelev", min_elev2)
        triangulation_app.SetParameterFloat("rightmaxelev", max_elev2)

        triangulation_app.Execute()

        llh = np.copy(triangulation_app.GetVectorImageAsNumpyArray("out"))

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
        # save os env
        env_save = os.environ.copy()

        if "OTB_GEOID_FILE" in os.environ:
            logging.warning(
                "The OTB_GEOID_FILE environment variable is set by the user,"
                " it might disturbed the OTBGeometry geoid management"
            )
            del os.environ["OTB_GEOID_FILE"]

        # create OTB application
        img1 = cars_conf[input_parameters.IMG1_TAG]
        img2 = cars_conf[input_parameters.IMG2_TAG]

        stereo_app = otbApplication.Registry.CreateApplication(
            "StereoRectificationGridGenerator"
        )

        stereo_app.SetParameterString("io.inleft", img1)
        stereo_app.SetParameterString("io.inright", img2)
        stereo_app.SetParameterInt("epi.step", epipolar_step)
        if dem is not None:
            stereo_app.SetParameterString("epi.elevation.dem", dem)
        if default_alt is not None:
            stereo_app.SetParameterFloat("epi.elevation.default", default_alt)
        if geoid is not None:
            stereo_app.SetParameterString("epi.elevation.geoid", geoid)
        stereo_app.Execute()
        # OTB doesn't do a new line, force it for next logger seen by user
        print("\n")
        # Export grids to numpy
        left_grid_as_array = np.copy(
            stereo_app.GetVectorImageAsNumpyArray("io.outleft")
        )
        right_grid_as_array = np.copy(
            stereo_app.GetVectorImageAsNumpyArray("io.outright")
        )

        epipolar_size_x, epipolar_size_y, baseline = (
            stereo_app.GetParameterInt("epi.rectsizex"),
            stereo_app.GetParameterInt("epi.rectsizey"),
            stereo_app.GetParameterFloat("epi.baseline"),
        )

        origin = stereo_app.GetImageOrigin("io.outleft")
        spacing = stereo_app.GetImageSpacing("io.outleft")

        # Convert epipolar size depending on the pixel size
        # TODO: remove this patch when OTB issue
        # https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/issues/2176
        # is resolved
        with rio.open(img1, "r") as rio_dst:
            pixel_size_x, pixel_size_y = (
                rio_dst.transform[0],
                rio_dst.transform[4],
            )

        mean_size = (abs(pixel_size_x) + abs(pixel_size_y)) / 2
        epipolar_size_x = int(np.floor(epipolar_size_x * mean_size))
        epipolar_size_y = int(np.floor(epipolar_size_y * mean_size))

        # we want disp_to_alt_ratio = resolution/(B/H), in m.pixel^-1
        disp_to_alt_ratio = 1 / baseline

        # restore environment variables
        if "OTB_GEOID_FILE" in env_save.keys():
            os.environ["OTB_GEOID_FILE"] = env_save["OTB_GEOID_FILE"]

        return (
            left_grid_as_array,
            right_grid_as_array,
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

        Be careful: When SRTM is used, the default elevation (altitude)
        doesn't work anymore (OTB function) when ConvertSensorToGeoPointFast
        is called again. Check the values.

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
        # save os env
        env_save = os.environ.copy()

        if "OTB_GEOID_FILE" in os.environ:
            logging.warning(
                "The OTB_GEOID_FILE environment variable is set by the user,"
                " it might disturbed the OTBGeometry geoid management"
            )
            del os.environ["OTB_GEOID_FILE"]

        # create OTB application
        img = cars_conf[
            input_parameters.create_img_tag_from_product_key(product_key)
        ]

        s2c_app = otbApplication.Registry.CreateApplication(
            "ConvertSensorToGeoPointFast"
        )

        s2c_app.SetParameterString("in", img)
        s2c_app.SetParameterFloat("input.idx", x_coord)
        s2c_app.SetParameterFloat("input.idy", y_coord)

        if z_coord is not None:
            s2c_app.SetParameterFloat("input.idz", z_coord)
        if dem is not None:
            s2c_app.SetParameterString("elevation.dem", dem)
        if geoid is not None:
            s2c_app.SetParameterString("elevation.geoid", geoid)
        if default_elevation is not None:
            s2c_app.SetParameterFloat("elevation.default", default_elevation)
        # else ConvertSensorToGeoPointFast have only X, Y and OTB

        s2c_app.Execute()

        lon = s2c_app.GetParameterFloat("output.idx")
        lat = s2c_app.GetParameterFloat("output.idy")
        alt = s2c_app.GetParameterFloat("output.idz")

        # restore environment variables
        if "OTB_GEOID_FILE" in env_save.keys():
            os.environ["OTB_GEOID_FILE"] = env_save["OTB_GEOID_FILE"]

        return np.array([lat, lon, alt])

    def image_envelope(
        self,
        conf,
        product_key: str,
        shp,
        dem=None,
        default_alt=None,
        geoid=None,
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
        # save os env
        env_save = os.environ.copy()

        if "OTB_GEOID_FILE" in os.environ:
            logging.warning(
                "The OTB_GEOID_FILE environment variable is set by the user,"
                " it might disturbed the OTBGeometry geoid management"
            )
            del os.environ["OTB_GEOID_FILE"]

        # create OTB application
        img = conf[
            input_parameters.create_img_tag_from_product_key(product_key)
        ]

        # reset OTB DEMHandler
        loc_app = otbApplication.Registry.CreateApplication(
            "ConvertSensorToGeoPointFast"
        )
        loc_app.SetParameterString("in", img)
        loc_app.SetParameterFloat("input.idx", 0.0)
        loc_app.SetParameterFloat("input.idy", 0.0)
        loc_app.SetParameterFloat("input.idz", 0.0)
        loc_app.Execute()

        app = otbApplication.Registry.CreateApplication("ImageEnvelope")

        if isinstance(img, str):
            app.SetParameterString("in", img)
        else:
            app.SetParameterInputImage("in", img)

        if dem is not None:
            app.SetParameterString("elev.dem", dem)

        if default_alt is not None:
            app.SetParameterFloat("elev.default", default_alt)

        if geoid is not None:
            app.SetParameterString("elev.geoid", geoid)

        app.SetParameterString("out", shp)
        app.ExecuteAndWriteOutput()

        # restore environment variables
        if "OTB_GEOID_FILE" in env_save.keys():
            os.environ["OTB_GEOID_FILE"] = env_save["OTB_GEOID_FILE"]
