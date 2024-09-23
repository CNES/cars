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
this module contains the bulldozer dsm filling application class.
"""

import os

import numpy as np
import rasterio as rio
import yaml
from bulldozer.pipeline.bulldozer_pipeline import dsm_to_dtm
from json_checker import Checker
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely import Polygon
from shareloc.dtm_reader import interpolate_geoid_height

from cars.core import inputs, projection

from . import dsm_filling_tools as dft
from .dsm_filling import DsmFilling


class BulldozerFilling(DsmFilling, short_name="bulldozer"):
    """
    BulldozerFilling
    """

    def __init__(self, conf=None):
        """
        Init function of BulldozerFilling

        :param conf: configuration for BulldozerFilling
        :return: an application_to_use object
        """
        super().__init__(conf=conf)

        # check conf
        self.used_method = self.used_config["method"]
        self.activated = self.used_config["activated"]

        # Init orchestrator
        self.orchestrator = None

    def check_conf(self, conf):

        # init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        # Overload conf
        overloaded_conf["method"] = conf.get("method", "bulldozer")
        overloaded_conf["activated"] = conf.get("activated", False)

        rectification_schema = {
            "method": str,
            "activated": bool,
        }

        # Check conf
        checker = Checker(rectification_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(
        self,
        initial_elevation,
        dsm_path,
        list_sensor_pairs_or_roi_poly,
        output_geoid,
        out_dir,
    ):
        """
        Run dsm filling using initial elevation and the current dsm
        Outputs the filled dsm in a new file, and the filling mask

        list_sensor_pairs_or_roi_poly can any of these objects :
            - a list of pairs of input images
            - a list of Shapely Polygons
            - a Shapely Polygon
        """

        if not self.activated:
            return

        temp_dsm_path = os.path.join(out_dir, "temp_dsm.tif")
        final_dsm_path = os.path.join(out_dir, "dsm_filled.tif")
        filling_path = os.path.join(out_dir, "filling_bulldozer.tif")

        # create the config for the bulldozer execution
        bull_conf_path = os.path.join(
            os.path.dirname(__file__), "bulldozer_config/base_config.yaml"
        )
        with open(bull_conf_path, "r", encoding="utf8") as bull_conf_file:
            bull_conf = yaml.safe_load(bull_conf_file)

        bull_conf["dsm_path"] = temp_dsm_path
        bull_conf["output_dir"] = os.path.join(out_dir, "bulldozer")
        bull_conf_path = os.path.join(out_dir, "bulldozer_config.yaml")
        with open(bull_conf_path, "w", encoding="utf8") as bull_conf_file:
            yaml.dump(bull_conf, bull_conf_file)

        dtm_path = os.path.join(bull_conf["output_dir"], "DTM.tif")

        # get dsm to be filled and its metadata
        with rio.open(dsm_path) as in_dsm:
            dsm = in_dsm.read()
            dsm_msk = in_dsm.read_masks()
            dsm_tr = in_dsm.transform
            dsm_crs = in_dsm.crs
            dsm_tr_mat = np.array(
                [
                    [dsm_tr[0], dsm_tr[1], dsm_tr[2]],
                    [dsm_tr[3], dsm_tr[4], dsm_tr[5]],
                    [0, 0, 1],
                ]
            )
            dsm_meta = in_dsm.meta

        roi_raster = np.ones(dsm[0].shape)
        polys = []
        if list_sensor_pairs_or_roi_poly.isinstance(list):
            for obj in list_sensor_pairs_or_roi_poly:
                if obj.isinstance(Polygon):
                    polys.append(obj)
                else:
                    (pair_key, _, _) = obj
                    pair_folder = os.path.join(out_dir, pair_key)
                    pair_poly, epsg = inputs.read_vector(
                        os.path.join(pair_folder, "envelopes_intersection.gpkg")
                    )
                    pair_poly = projection.polygon_projection(
                        pair_poly, epsg, dsm_crs.to_epsg()
                    )
                    polys.append(pair_poly)
        elif list_sensor_pairs_or_roi_poly.isinstance(Polygon):
            polys.append(list_sensor_pairs_or_roi_poly)

        if len(polys) > 0:
            roi_raster = rio.features.rasterize(
                polys, out_shape=roi_raster.shape, transform=dsm_tr
            )

        # get the initial elevation
        with rio.open(initial_elevation.dem) as in_elev:

            elev_tr = in_elev.transform
            elev_crs = in_elev.crs
            elev_tr_mat = np.array(
                [
                    [elev_tr[0], elev_tr[1], elev_tr[2]],
                    [elev_tr[3], elev_tr[4], elev_tr[5]],
                    [0, 0, 1],
                ]
            )
            elev_tr_inv_mat = np.linalg.inv(elev_tr_mat)

            # get reading window
            ijs_window = np.array(
                [
                    [0, 0],
                    [dsm.shape[1], dsm.shape[2]],
                ]
            )
            # third: project elev crs to elev pixel coords
            ijs_window_dem = dft.project(
                # second: project dsm crs to elev crs
                projection.points_cloud_conversion(
                    # first: project dsm pixel coords to dsm crs
                    dft.project(ijs_window[:, [1, 0]], dsm_tr_mat),
                    dsm_crs.to_epsg(),
                    elev_crs.to_epsg(),
                ),
                elev_tr_inv_mat,
            )[:, [1, 0]]

            # get initial elevation
            elev = in_elev.read(
                out_shape=(1,) + dsm.shape[1:],
                window=Window.from_slices(
                    (ijs_window_dem[0, 0], ijs_window_dem[1, 0]),
                    (ijs_window_dem[0, 1], ijs_window_dem[1, 1]),
                ),
                resampling=Resampling.bilinear,
            )

        temp_filled_dsm = dsm.copy()
        temp_filled_dsm[dsm_msk == 0] = elev[dsm_msk == 0]

        # apply offset to project on geoid if needed
        if output_geoid is not True:
            to_fill_ijs = np.argwhere(dsm_msk[0] == 0)
            to_fill_xys = dft.project(to_fill_ijs[:, [1, 0]], dsm_tr_mat)

            applied_offset = np.zeros(len(to_fill_ijs))  # (n,)
            pts_to_project_on_geoid = projection.points_cloud_conversion(
                to_fill_xys, dsm_crs.to_epsg(), 4326
            )

            if output_geoid.isinstance(bool) and output_geoid is False:
                # out geoid is ellipsoid: add geoid-ellipsoid distance
                applied_offset += interpolate_geoid_height(
                    initial_elevation.geoid, pts_to_project_on_geoid
                )
            elif output_geoid.isinstance(str):
                # out geoid is a new geoid whose path is in output_geoid:
                # add carsgeoid-ellipsoid distance then add ellipsoid-outgeoid
                applied_offset += interpolate_geoid_height(
                    initial_elevation.geoid, pts_to_project_on_geoid
                )
                applied_offset -= interpolate_geoid_height(
                    output_geoid, pts_to_project_on_geoid
                )

            temp_filled_dsm[
                0, to_fill_ijs[:, 0], to_fill_ijs[:, 1]
            ] += applied_offset

        with rio.open(temp_dsm_path, "w", **dsm_meta) as out_dsm:
            out_dsm.write(temp_filled_dsm)

        dsm_to_dtm(bull_conf_path)

        with rio.open(dtm_path) as in_dtm:
            dtm = in_dtm.read()

        filling_mask = np.logical_and(dsm_msk == 0, roi_raster > 0)
        dsm[filling_mask] = dtm[filling_mask]

        with rio.open(final_dsm_path, "w", **dsm_meta) as out_dsm:
            out_dsm.write(dsm)

        with rio.open(filling_path, "w", **dsm_meta) as out_dsm:
            out_dsm.write(filling_mask)

        # remove the temporary DSM used to run Bulldozer
        os.remove(temp_dsm_path)
