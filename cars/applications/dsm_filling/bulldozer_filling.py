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
This module contains the bulldozer dsm filling application class.
"""

import contextlib
import logging
import os
import shutil

import numpy as np
import rasterio as rio
import yaml
from bulldozer.pipeline.bulldozer_pipeline import dsm_to_dtm
from json_checker import Checker
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely import Polygon
from shareloc.dtm_reader import interpolate_geoid_height

from cars.core import projection

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
        self.save_intermediate_data = self.used_config["save_intermediate_data"]

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
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        rectification_schema = {
            "method": str,
            "activated": bool,
            "save_intermediate_data": bool,
        }

        # Check conf
        checker = Checker(rectification_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(  # noqa C901
        self,
        orchestrator,
        initial_elevation,
        dsm_path,
        roi_polys,
        roi_epsg,
        output_geoid,
        filling_file_name,
        dump_dir,
    ):
        """
        Run dsm filling using initial elevation and the current dsm
        Replaces dsm.tif by the filled dsm. Adds a new band
        to filling.tif if it exists.
        The old dsm is saved in dump_dir.

        roi_poly can any of these objects :
            - a list of Shapely Polygons
            - a Shapely Polygon
        """

        if not self.activated:
            return

        if initial_elevation is None:
            logging.error("No DEM was provided, dsm_filling will not run.")
            return

        dump_dir = os.path.join(dump_dir, "dsm_filling")

        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        old_dsm_path = os.path.join(dump_dir, "dsm_not_filled.tif")
        temp_dsm_path = os.path.join(
            dump_dir, "dsm_filled_with_dem_not_smoothed.tif"
        )
        final_dsm_path = dsm_path

        # create the config for the bulldozer execution
        bull_conf_path = os.path.join(
            os.path.dirname(__file__), "bulldozer_config/base_config.yaml"
        )
        with open(bull_conf_path, "r", encoding="utf8") as bull_conf_file:
            bull_conf = yaml.safe_load(bull_conf_file)

        bull_conf["dsm_path"] = temp_dsm_path
        bull_conf["output_dir"] = os.path.join(dump_dir, "bulldozer")

        if orchestrator is not None:
            if (
                orchestrator.get_conf()["mode"] == "multiprocessing"
                or orchestrator.get_conf()["mode"] == "local_dask"
            ):
                bull_conf["nb_max_workers"] = orchestrator.get_conf()[
                    "nb_workers"
                ]

        bull_conf_path = os.path.join(dump_dir, "bulldozer_config.yaml")
        with open(bull_conf_path, "w", encoding="utf8") as bull_conf_file:
            yaml.dump(bull_conf, bull_conf_file)

        dtm_path = os.path.join(bull_conf["output_dir"], "dtm.tif")

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

        if isinstance(roi_polys, list):
            roi_polys_outepsg = []
            for poly in roi_polys:
                if isinstance(poly, Polygon):
                    roi_poly_outepsg = projection.polygon_projection(
                        poly, roi_epsg, dsm_crs.to_epsg()
                    )
                    roi_polys_outepsg.append(roi_poly_outepsg)

            roi_raster = rio.features.rasterize(
                roi_polys_outepsg, out_shape=roi_raster.shape, transform=dsm_tr
            )
        elif isinstance(roi_polys, Polygon):
            roi_poly_outepsg = projection.polygon_projection(
                roi_polys, roi_epsg, dsm_crs.to_epsg()
            )
            roi_raster = rio.features.rasterize(
                [roi_poly_outepsg], out_shape=roi_raster.shape, transform=dsm_tr
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
                projection.point_cloud_conversion(
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
            pts_to_project_on_geoid = projection.point_cloud_conversion(
                to_fill_xys, dsm_crs.to_epsg(), 4326
            )

            if isinstance(output_geoid, bool) and output_geoid is False:
                # out geoid is ellipsoid: add geoid-ellipsoid distance
                applied_offset += interpolate_geoid_height(
                    initial_elevation.geoid, pts_to_project_on_geoid
                )
            elif isinstance(output_geoid, str):
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

        try:
            try:
                # suppress prints in bulldozer by redirecting stdout&stderr
                with open(os.devnull, "w", encoding="utf8") as devnull:
                    with (
                        contextlib.redirect_stdout(devnull),
                        contextlib.redirect_stderr(devnull),
                    ):
                        dsm_to_dtm(bull_conf_path)
            except Exception:
                logging.info(
                    "Bulldozer failed on its first execution. Retrying"
                )
                # suppress prints in bulldozer by redirecting stdout&stderr
                with open(os.devnull, "w", encoding="utf8") as devnull:
                    with (
                        contextlib.redirect_stdout(devnull),
                        contextlib.redirect_stderr(devnull),
                    ):
                        dsm_to_dtm(bull_conf_path)
        except Exception:
            logging.error(
                "Bulldozer failed on its second execution."
                + " The DSM could not be filled."
            )
        else:
            with rio.open(dtm_path) as in_dtm:
                dtm = in_dtm.read()

            with rio.open(old_dsm_path, "w", **dsm_meta) as out_dsm:
                out_dsm.write(dsm)

            filling_mask = np.logical_and(dsm_msk == 0, roi_raster > 0)
            dsm[filling_mask] = dtm[filling_mask]

            with rio.open(final_dsm_path, "w", **dsm_meta) as out_dsm:
                out_dsm.write(dsm)

            if filling_file_name is not None:

                with rio.open(filling_file_name, "r") as src:
                    fill_meta = src.meta
                    bands = [src.read(i + 1) for i in range(src.count)]
                    bands_desc = [src.descriptions[i] for i in range(src.count)]

                fill_meta["count"] += 1

                bands.append(filling_mask[0].astype(np.uint8))
                bands_desc.append("filling_bulldozer")

                with rio.open(filling_file_name, "w", **fill_meta) as out:
                    for i, band in enumerate(bands):
                        out.write(band, i + 1)
                        out.set_band_description(i + 1, bands_desc[i])

            # reason for this to be indented: don't remove intermediate
            # files if bulldozer failed twice (for the logs)
            if not self.save_intermediate_data:
                # remove intermediary files if not needed
                try:
                    shutil.rmtree(dump_dir)
                except Exception as exception_rmtree:
                    logging_msg = (
                        "dsm_filling's intermediary data "
                        + "could not be deleted, "
                        + f"an error occured: {exception_rmtree}."
                    )
                    logging.info(logging_msg)
