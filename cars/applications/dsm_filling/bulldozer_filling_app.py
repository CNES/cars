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
from json_checker import Checker, Or
from pyproj import CRS
from shapely import Polygon

from cars.core import inputs, projection
from cars.orchestrator.cluster.log_wrapper import cars_profile

from .abstract_dsm_filling_app import DsmFilling


class BulldozerFilling(DsmFilling, short_name="bulldozer"):
    """
    Bulldozer filling
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
        self.classification = self.used_config["classification"]
        self.save_intermediate_data = self.used_config["save_intermediate_data"]

    def check_conf(self, conf):

        # init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        # Overload conf
        overloaded_conf["method"] = conf.get("method", "bulldozer")
        overloaded_conf["classification"] = conf.get("classification", "nodata")

        if isinstance(overloaded_conf["classification"], str):
            overloaded_conf["classification"] = [
                overloaded_conf["classification"]
            ]

        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        rectification_schema = {
            "method": str,
            "classification": Or(None, [str]),
            "save_intermediate_data": bool,
        }

        # Check conf
        checker = Checker(rectification_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    @cars_profile(name="Bulldozer filling")
    def run(  # pylint: disable=too-many-positional-arguments # noqa C901
        self,
        dsm_file,
        classif_file,
        filling_file,
        dump_dir,
        roi_polys,
        roi_epsg,
        orchestrator,
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

        if self.classification is None:
            self.classification = ["nodata"]

        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        old_dsm_path = os.path.join(dump_dir, "dsm_not_filled.tif")
        new_dsm_path = os.path.join(dump_dir, "dsm_filled.tif")

        # create the config for the bulldozer execution
        bull_conf_path = os.path.join(
            os.path.dirname(__file__), "bulldozer_config/base_config.yaml"
        )
        with open(bull_conf_path, "r", encoding="utf8") as bull_conf_file:
            bull_conf = yaml.safe_load(bull_conf_file)

        bull_conf["dsm_path"] = dsm_file
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
        with rio.open(dsm_file) as in_dsm:
            dsm = in_dsm.read(1)
            dsm_tr = in_dsm.transform
            dsm_crs = in_dsm.crs
            dsm_meta = in_dsm.meta

        roi_raster = np.ones(dsm.shape)

        if isinstance(roi_polys, list):
            roi_polys_outepsg = []
            for poly in roi_polys:
                if isinstance(poly, Polygon):
                    roi_poly_outepsg = projection.polygon_projection_crs(
                        poly, CRS(roi_epsg), dsm_crs
                    )
                    roi_polys_outepsg.append(roi_poly_outepsg)

            roi_raster = rio.features.rasterize(
                roi_polys_outepsg, out_shape=roi_raster.shape, transform=dsm_tr
            )
        elif isinstance(roi_polys, Polygon):
            roi_poly_outepsg = projection.polygon_projection_crs(
                roi_polys, CRS(roi_epsg), dsm_crs
            )
            roi_raster = rio.features.rasterize(
                [roi_poly_outepsg], out_shape=roi_raster.shape, transform=dsm_tr
            )

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
            logging.warning(
                "Bulldozer failed on its second execution."
                + " The DSM could not be filled."
            )
            return None
        with rio.open(dtm_path) as in_dtm:
            dtm = in_dtm.read(1)

        if self.save_intermediate_data:
            with rio.open(old_dsm_path, "w", **dsm_meta) as out_dsm:
                out_dsm.write(dsm, 1)

        if classif_file is not None and os.path.exists(classif_file):
            classif_descriptions = inputs.get_descriptions_bands(classif_file)
        else:
            classif_descriptions = []
        combined_mask = np.zeros_like(dsm).astype(np.uint8)
        for label in self.classification:
            if label in classif_descriptions:
                index_classif = classif_descriptions.index(label) + 1
                with rio.open(classif_file) as in_classif:
                    classif = in_classif.read(index_classif)
                    classif_msk = in_classif.read_masks(1)
                classif[classif_msk == 0] = 0
                filling_mask = np.logical_and(classif, roi_raster > 0)
            elif label == "nodata":
                if classif_file is not None and os.path.exists(classif_file):
                    with rio.open(classif_file) as in_classif:
                        classif_msk = in_classif.read_masks(1)
                    classif = ~classif_msk
                else:
                    with rio.open(dsm_file) as in_dsm:
                        dsm_msk = in_dsm.read_masks(1)
                    classif = ~dsm_msk
                filling_mask = np.logical_and(classif, roi_raster > 0)
            else:
                logging.error(
                    "Label {} not found in classification "
                    "descriptions {}".format(label, classif_descriptions)
                )
                continue
            logging.info("Filling of {} with Bulldozer DTM".format(label))
            dsm[filling_mask] = dtm[filling_mask]
            combined_mask = np.logical_or(combined_mask, filling_mask)

        with rio.open(dsm_file, "w", **dsm_meta) as out_dsm:
            out_dsm.write(dsm, 1)
        if self.save_intermediate_data:
            shutil.copy2(dsm_file, new_dsm_path)

        if filling_file is not None:
            with rio.open(filling_file, "r") as src:
                fill_meta = src.meta
                bands = [src.read(i + 1) for i in range(src.count)]
                bands_desc = [src.descriptions[i] for i in range(src.count)]
            fill_meta["count"] += 1
            bands.append(combined_mask.astype(np.uint8))
            bands_desc.append("bulldozer")

            with rio.open(filling_file, "w", **fill_meta) as out:
                for i, band in enumerate(bands):
                    out.write(band, i + 1)
                    out.set_band_description(i + 1, bands_desc[i])
        return dtm_path
