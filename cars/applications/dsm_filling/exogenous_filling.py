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

import logging
import os

import numpy as np
import rasterio as rio
from json_checker import Checker, Or
from rasterio.enums import Resampling
from rasterio.warp import reproject
from shapely import Polygon
from shareloc.dtm_reader import interpolate_geoid_height

from cars.core import inputs, projection

from . import dsm_filling_tools as dft
from .dsm_filling import DsmFilling


class ExogenousFilling(DsmFilling, short_name="exogenous_filling"):
    """
    Exogenous filling
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
        self.classification = self.used_config["classification"]
        self.fill_with_geoid = self.used_config["fill_with_geoid"]
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
        overloaded_conf["fill_with_geoid"] = conf.get("fill_with_geoid", None)
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        rectification_schema = {
            "method": str,
            "activated": bool,
            "classification": Or(None, [str]),
            "fill_with_geoid": Or(None, [str]),
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
        dump_dir,
        roi_polys,
        roi_epsg,
        output_geoid,
        geom_plugin,
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

        if self.fill_with_geoid is None:
            self.fill_with_geoid = []

        if geom_plugin is None:
            logging.error(
                "No DEM was provided, exogenous_filling will not run."
            )
            return

        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        old_dsm_path = os.path.join(dump_dir, "dsm_not_filled.tif")
        new_dsm_path = os.path.join(dump_dir, "dsm_filled.tif")

        # get dsm to be filled and its metadata
        with rio.open(dsm_file) as in_dsm:
            dsm = in_dsm.read(1)
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

        # get the initial elevation
        with rio.open(geom_plugin.dem) as in_elev:
            # Reproject the elevation data to match the DSM
            elev_data = np.empty(dsm.shape, dtype=in_elev.dtypes[0])

            reproject(
                source=rio.band(in_elev, 1),
                destination=elev_data,
                src_transform=in_elev.transform,
                src_crs=in_elev.crs,
                dst_transform=dsm_tr,
                dst_crs=dsm_crs,
                resampling=Resampling.bilinear,
            )

        # Save old dsm
        with rio.open(old_dsm_path, "w", **dsm_meta) as out_dsm:
            out_dsm.write(dsm, 1)

        # Fill DSM for every label
        combined_mask = np.zeros_like(dsm).astype(np.uint8)
        if classif_file is not None:
            classif_descriptions = inputs.get_descriptions_bands(classif_file)
        else:
            classif_descriptions = []
        for label in self.classification:
            if label in classif_descriptions:
                index_classif = classif_descriptions.index(label) + 1
                with rio.open(classif_file) as in_classif:
                    classif = in_classif.read(index_classif)
                    classif_msk = in_classif.read_masks(1)
                classif[classif_msk == 0] = 0
                filling_mask = np.logical_and(classif, roi_raster > 0)
            elif label == "nodata":
                if classif_file is not None:
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

            if label in self.fill_with_geoid:
                logging.info("Filling of {} with geoid".format(label))
                dsm[filling_mask] = 0
            else:
                logging.info("Filling of {} with DEM and geoid".format(label))
                dsm[filling_mask] = elev_data[filling_mask]

            # apply offset to project on geoid if needed
            if output_geoid is not True:
                to_fill_ijs = np.argwhere(filling_mask)
                to_fill_xys = dft.project(to_fill_ijs[:, [1, 0]], dsm_tr_mat)

                applied_offset = np.zeros(len(to_fill_ijs))  # (n,)
                pts_to_project_on_geoid = projection.point_cloud_conversion(
                    to_fill_xys, dsm_crs.to_epsg(), 4326
                )

                if isinstance(output_geoid, bool) and output_geoid is False:
                    # out geoid is ellipsoid: add geoid-ellipsoid distance
                    applied_offset += interpolate_geoid_height(
                        geom_plugin.geoid, pts_to_project_on_geoid
                    )
                elif isinstance(output_geoid, str):
                    # out geoid is a new geoid whose path is in output_geoid:
                    # add carsgeoid-ellipsoid then add ellipsoid-outgeoid
                    applied_offset += interpolate_geoid_height(
                        geom_plugin.geoid, pts_to_project_on_geoid
                    )
                    applied_offset -= interpolate_geoid_height(
                        output_geoid, pts_to_project_on_geoid
                    )

                dsm[filling_mask] += applied_offset
            combined_mask = np.logical_or(combined_mask, filling_mask)

        with rio.open(new_dsm_path, "w", **dsm_meta) as out_dsm:
            out_dsm.write(dsm, 1)
        with rio.open(dsm_file, "w", **dsm_meta) as out_dsm:
            out_dsm.write(dsm, 1)

        if filling_file is not None:
            with rio.open(filling_file, "r") as src:
                fill_meta = src.meta
                bands = [src.read(i + 1) for i in range(src.count)]
                bands_desc = [src.descriptions[i] for i in range(src.count)]
            fill_meta["count"] += 1
            bands.append(combined_mask)
            bands_desc.append("filling_exogenous")

            with rio.open(filling_file, "w", **fill_meta) as out:
                for i, band in enumerate(bands):
                    out.write(band, i + 1)
                    out.set_band_description(i + 1, bands_desc[i])
