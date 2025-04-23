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
This module contains the border interpolation dsm filling application class.
"""

import logging
import os
import shutil

import numpy as np
import rasterio as rio
import scipy
import skimage
from json_checker import Checker, Or
from shapely import Polygon

from cars.core import inputs, projection

from .dsm_filling import DsmFilling


class BorderInterpolation(DsmFilling, short_name="border_interpolation"):
    """
    Border interpolation
    """

    def __init__(self, conf=None):
        """
        Init function of BorderInterpolation

        :param conf: configuration for BulldozerFilling
        :return: an application_to_use object
        """
        super().__init__(conf=conf)

        # check conf
        self.used_method = self.used_config["method"]
        self.activated = self.used_config["activated"]
        self.classification = self.used_config["classification"]
        self.component_min_size = self.used_config["component_min_size"]
        self.border_size = self.used_config["border_size"]
        self.percentile = self.used_config["percentile"]
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
        overloaded_conf["activated"] = conf.get("activated", False)
        overloaded_conf["classification"] = conf.get("classification", None)
        overloaded_conf["component_min_size"] = conf.get(
            "component_min_size", 5
        )
        overloaded_conf["border_size"] = conf.get("border_size", 10)
        overloaded_conf["percentile"] = conf.get("percentile", 10)
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        rectification_schema = {
            "method": str,
            "activated": bool,
            "classification": Or(None, [str]),
            "component_min_size": int,
            "border_size": int,
            "percentile": Or(int, float),
            "save_intermediate_data": bool,
        }

        # Check conf
        checker = Checker(rectification_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(  # noqa C901
        self,
        dsm_file,
        classif_file,
        filling_file,
        dtm_file,
        dump_dir,
        roi_polys,
        roi_epsg,
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

        if self.classification is None:
            self.classification = ["nodata"]
            logging.error(
                "Filling method 'border_interpolation' needs a classification"
            )

        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        old_dsm_path = os.path.join(dump_dir, "dsm_not_filled.tif")
        new_dsm_path = os.path.join(dump_dir, "dsm_filled.tif")

        # get dsm to be filled and its metadata
        with rio.open(dsm_file) as in_dsm:
            dsm = in_dsm.read(1)
            dsm_tr = in_dsm.transform
            dsm_crs = in_dsm.crs
            dsm_meta = in_dsm.meta
            dsm_nodata = in_dsm.nodata

        roi_raster = np.ones(dsm.shape)

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

        # get dtm to fill the dsm
        if dtm_file is not None:
            logging.info(
                "Use DTM file {} for border interpolation".format(dtm_file)
            )
            with rio.open(dtm_file) as in_dtm:
                dtm = in_dtm.read(1)
                dtm_nodata = in_dtm.nodata
        else:
            logging.info(
                "No DTM provided : DSM {} will be used for "
                "border interpolation".format(dsm_file)
            )
            dtm = dsm.copy()
            dtm_nodata = dsm_nodata
        dtm[dtm == dtm_nodata] = np.nan

        if self.save_intermediate_data:
            with rio.open(old_dsm_path, "w", **dsm_meta) as out_dsm:
                out_dsm.write(dsm, 1)

        if classif_file is not None:
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
            else:
                logging.error(
                    "Label {} not found in classification "
                    "descriptions {}".format(label, classif_descriptions)
                )
                continue
            logging.info(
                "Filling of {} with Bulldozer DTM using "
                "border interpolation".format(label)
            )
            filling_mask[classif_msk == 0] = 0
            filling_mask = skimage.morphology.binary_opening(
                filling_mask,
                footprint=[
                    (np.ones((self.component_min_size, 1)), 1),
                    (np.ones((1, self.component_min_size)), 1),
                ],
            )
            features, num_features = scipy.ndimage.label(filling_mask)
            logging.info("Filling of {} features".format(num_features))
            features_boundaries = skimage.morphology.dilation(
                features,
                footprint=[
                    (np.ones((self.border_size, 1)), 1),
                    (np.ones((1, self.border_size)), 1),
                ],
            )
            features_boundaries[filling_mask] = 0
            borders_file_path = os.path.join(
                dump_dir, "borders_of_{}.tif".format(label)
            )
            if self.save_intermediate_data:
                with rio.open(
                    borders_file_path, "w", **dsm_meta
                ) as out_borders:
                    out_borders.write(features_boundaries, 1)
            for feature_id in range(1, num_features + 1):
                altitude = np.nanpercentile(
                    dtm[features_boundaries == feature_id], self.percentile
                )
                if altitude is not None:
                    dsm[features == feature_id] = altitude
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
            bands_desc.append("border_interpolation")

            with rio.open(filling_file, "w", **fill_meta) as out:
                for i, band in enumerate(bands):
                    out.write(band, i + 1)
                    out.set_band_description(i + 1, bands_desc[i])
