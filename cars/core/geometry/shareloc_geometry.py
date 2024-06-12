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
from typing import List, Tuple, Union

import bindings_cpp
import numpy as np
import rasterio as rio
import shareloc.geofunctions.rectification as rectif
import xarray as xr
from json_checker import Checker
from shareloc.dtm_reader import dtm_reader
from shareloc.geofunctions import localization
from shareloc.geofunctions.triangulation import epipolar_triangulation
from shareloc.geomodels.geomodel import GeoModel
from shareloc.geomodels.grid import Grid
from shareloc.geomodels.rpc import RPC
from shareloc.image import Image

from cars.core import constants as cst
from cars.core import inputs, projection
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.data_structures import cars_dataset

GRID_TYPE = "GRID"
RPC_TYPE = "RPC"
GEO_MODEL_PATH_TAG = "path"
GEO_MODEL_TYPE_TAG = "model_type"


@AbstractGeometry.register_subclass("SharelocGeometry")
class SharelocGeometry(AbstractGeometry):
    """
    Shareloc geometry class
    """

    def __init__(
        self,
        geometry_plugin,
        dem=None,
        geoid=None,
        default_alt=None,
        pairs_for_roi=None,
    ):
        super().__init__(
            geometry_plugin,
            dem=dem,
            geoid=geoid,
            default_alt=default_alt,
            pairs_for_roi=pairs_for_roi,
        )

        self.dem_roi = None
        self.roi_shareloc = None
        self.elevation = None

        # compute roi only when generating geometry object with dem
        # even if dem is None
        if geoid is not None and pairs_for_roi is not None:
            self.dem_roi_epsg = 4326
            if dem is not None:
                # Get dem epsg
                self.dem_roi_epsg = inputs.rasterio_get_epsg(dem)

            self.roi_shareloc = self.get_roi(
                pairs_for_roi, self.dem_roi_epsg, margin=0.012
            )
            # change convention
            self.dem_roi = [
                self.roi_shareloc[1],
                self.roi_shareloc[0],
                self.roi_shareloc[3],
                self.roi_shareloc[2],
            ]

        if dem is not None:

            # fill_nodata option should be set when dealing with void in DTM
            # see shareloc DTM limitations in sphinx doc for further details
            dtm_image = dtm_reader(
                dem,
                geoid,
                roi=self.roi_shareloc,
                roi_is_in_physical_space=True,
                fill_nodata="mean",
                fill_value=0.0,
            )
            self.elevation = (
                bindings_cpp.DTMIntersection(  # pylint: disable=I1101
                    dtm_image.epsg,
                    dtm_image.alt_data,
                    dtm_image.nb_rows,
                    dtm_image.nb_columns,
                    dtm_image.transform,
                )
            )
        else:
            self.elevation = default_alt

    def get_roi(self, pairs_for_roi, epsg, margin=0.006):
        """
        Compute region of interest for intersection of DEM

        :param pairs_for_roi: list of pairs of images and geomodels
        :type pairs_for_roi: List[(str, dict, str, dict)]
        :param dem_epsg: output EPSG code for ROI
        :type dem_epsg: int
        :param margin: margin for ROI in degrees
        :type margin: float
        """
        coords_list = []
        for image1, geomodel1, image2, geomodel2 in pairs_for_roi:
            # Footprint of left image
            coords_list.extend(self.image_envelope(image1, geomodel1))
            # Footprint of right image
            coords_list.extend(self.image_envelope(image2, geomodel2))
            # Epipolar extent
            image1 = SharelocGeometry.load_image(image1)
            geomodel1 = self.load_geom_model(geomodel1)
            geomodel2 = self.load_geom_model(geomodel2)
            epipolar_extent = rectif.get_epipolar_extent(
                image1, geomodel1, geomodel2
            )
            lat_min, lon_min, lat_max, lon_max = list(epipolar_extent)
            coords_list.extend([(lon_min, lat_min), (lon_max, lat_max)])

        lon_list, lat_list = list(zip(*coords_list))  # noqa: B905
        roi = [
            min(lat_list) - margin,
            min(lon_list) - margin,
            max(lat_list) + margin,
            max(lon_list) + margin,
        ]
        points = np.array(
            [
                (roi[1], roi[0], 0),
                (roi[3], roi[2], 0),
                (roi[1], roi[0], 0),
                (roi[3], roi[2], 0),
            ]
        )
        new_points = projection.points_cloud_conversion(points, 4326, epsg)
        roi = [
            min(new_points[:, 1]),
            min(new_points[:, 0]),
            max(new_points[:, 1]),
            max(new_points[:, 0]),
        ]
        return roi

    @staticmethod
    def load_geom_model(model: dict) -> Union[Grid, RPC]:
        """
        Load geometric model and returns it as a shareloc object

        :param model: Path and attributes for geometrical model
        :type model: dict with keys "path" and "model_type"
        :return: geometric model as a shareloc object (Grid or RPC)
        """
        geomodel = model[GEO_MODEL_PATH_TAG]
        # Use RPC Type if none are used
        if model.get(GEO_MODEL_TYPE_TAG):
            geomodel_type = model[GEO_MODEL_TYPE_TAG]
        else:
            geomodel_type = RPC_TYPE

        # Use RPCoptim class to use optimized C++ direct localizations
        if geomodel_type == "RPC":
            geomodel_type = "RPCoptim"

        shareloc_model = GeoModel(geomodel, geomodel_type)

        if shareloc_model is None:
            raise ValueError(f"Model {geomodel} could not be read by shareloc")

        return shareloc_model

    @staticmethod
    def load_image(img: str) -> Image:
        """
        Load the image using the Image class of Shareloc

        :param img: path to the image
        :return: The Image object
        """
        try:
            shareloc_img = Image(img, vertical_direction="north")
        except Exception as error:
            raise ValueError(f"Image type {img} is not supported") from error

        return shareloc_img

    @staticmethod
    def check_product_consistency(sensor: str, geomodel: dict) -> bool:
        """
        Test if the product is readable by the shareloc plugin

        TODO: not used
        - to evolve and use in CARS configuration early in pipeline process
        (new early check input common application ?)

        :param sensor: path to sensor image
        :param geomodel: path and attributes for geometrical model
        :return: sensor path and overloaded geomodel dict
        """
        # Check geomodel schema consistency
        if isinstance(geomodel, str):
            geomodel = {
                "path": geomodel,
            }
        overloaded_geomodel = geomodel.copy()

        # If model_type is not defined, default is "RPC"
        overloaded_geomodel["model_type"] = geomodel.get("model_type", "RPC")

        geomodel_schema = {"path": str, "model_type": str}
        checker_geomodel = Checker(geomodel_schema)
        checker_geomodel.validate(overloaded_geomodel)

        # Try to read them using shareloc
        SharelocGeometry.load_image(sensor)
        SharelocGeometry.load_geom_model(overloaded_geomodel)

        return sensor, overloaded_geomodel

    @staticmethod
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
        :param grid1: path or dataset for epipolar grid of sensor1
        :param grid2: path or dataset for epipolar grid of sensor2
        :param roi_key: dataset roi to use
        (can be cst.ROI or cst.ROI_WITH_MARGINS)

        :return: the long/lat/height numpy array in output of the triangulation
        """
        # read geomodels using shareloc
        shareloc_model1 = SharelocGeometry.load_geom_model(geomodel1)
        shareloc_model2 = SharelocGeometry.load_geom_model(geomodel2)

        # get path if grid is of type CarsDataset TODO remove
        if isinstance(grid1, cars_dataset.CarsDataset):
            grid1 = grid1.attributes["path"]
        if isinstance(grid2, cars_dataset.CarsDataset):
            grid2 = grid2.attributes["path"]

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
                "shareloc plugin triangulation".format(mode)
            )
            raise ValueError(
                f"{mode} mode is not available"
                " in the shareloc plugin triangulation"
            )

        return llh

    def generate_epipolar_grids(
        self,
        sensor1,
        sensor2,
        geomodel1,
        geomodel2,
        epipolar_step: int = 30,
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
            - the epipolar image size as a list of int
            (x-axis size is given with the index 0, y-axis size with index 1)
            - the disparity to altitude ratio as a float

        """
        # read inputs using shareloc
        shareloc_model1 = SharelocGeometry.load_geom_model(geomodel1)
        shareloc_model2 = SharelocGeometry.load_geom_model(geomodel2)

        image1 = SharelocGeometry.load_image(sensor1)
        image2 = SharelocGeometry.load_image(sensor2)

        # compute epipolar grids
        (
            grid1,
            grid2,
            [epipolar_size_y, epipolar_size_x],
            alt_to_disp_ratio,
            _,
        ) = rectif.compute_stereorectification_epipolar_grids(
            image1,
            shareloc_model1,
            image2,
            shareloc_model2,
            self.elevation,
            epi_step=epipolar_step,
        )

        # rearrange output to match the expected structure of CARS
        # grid[:, :, 2] with altitudes is not used
        grid1 = grid1[:, :, 0:2][:, :, ::-1]
        grid2 = grid2[:, :, 0:2][:, :, ::-1]

        # compute associated characteristics
        with rio.open(sensor1, "r") as rio_dst:
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

    def direct_loc(
        self,
        sensor,
        geomodel,
        x_coord: np.array,
        y_coord: np.array,
        z_coord: np.array = None,
    ) -> np.ndarray:
        """
        For a given image point, compute the latitude, longitude, altitude

        Advice: to be sure, use x,y,z inputs only

        :param sensor: path to sensor image
        :param geomodel: path and attributes for geomodel
        :param x_coord: X Coordinate in input image sensor
        :param y_coord: Y Coordinate in input image sensor
        :param z_coord: Z Altitude coordinate to take the image
        :return: Latitude, Longitude, Altitude coordinates as a numpy array
        """
        # load model and image with shareloc
        shareloc_model = SharelocGeometry.load_geom_model(geomodel)
        shareloc_image = SharelocGeometry.load_image(sensor)

        # perform direct localization operation
        loc = localization.Localization(
            shareloc_model,
            image=shareloc_image,
            elevation=self.elevation,
            epsg=4326,
        )
        # Bug: y_coord and x_coord inversion to fit Shareloc standards row/col.
        # TODO: clean geometry convention calls in API
        lonlatalt = loc.direct(
            y_coord, x_coord, z_coord, using_geotransform=True
        )
        lonlatalt = np.squeeze(lonlatalt)
        if len(lonlatalt.shape) == 1:
            latlonalt = np.array([lonlatalt[1], lonlatalt[0], lonlatalt[2]])
        else:
            latlonalt = np.array(
                [lonlatalt[:, 1], lonlatalt[:, 0], lonlatalt[:, 2]]
            )
        return latlonalt

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

        # load model and image with shareloc
        shareloc_model = SharelocGeometry.load_geom_model(geomodel)
        shareloc_image = SharelocGeometry.load_image(sensor)

        # perform inverse localization operation
        loc = localization.Localization(
            shareloc_model,
            image=shareloc_image,
            elevation=self.elevation,
            epsg=4326,
        )
        # Rows and columns order is inversed
        col, row, alti = loc.inverse(
            lon_coord.astype(np.float64),
            lat_coord.astype(np.float64),
            h=z_coord.astype(np.float64),
            using_geotransform=True,
        )

        return row, col, alti
