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
This module contains the interpolation dsm filling application class.
"""

import logging
import os
import shutil

import numpy as np
import rasterio as rio
import xarray as xr
from json_checker import Checker, Or
from pyproj import CRS
from rasterio.windows import Window
from rasterio.windows import transform as row_col_to_coords
from scipy import ndimage
from shapely import Polygon

import cars.orchestrator.orchestrator as ocht
from cars.core import projection
from cars.data_structures import cars_dataset
from cars.orchestrator.cluster.log_wrapper import cars_profile

from .abstract_dsm_filling_app import DsmFilling


class InterpolationFilling(DsmFilling, short_name="interpolation"):
    """
    Interpolation filling.
    """

    def __init__(self, conf=None):
        """
        Init function of InterpolationFilling.

        :param conf: configuration for InterpolationFilling
        :return: an application_to_use object
        """
        super().__init__(conf=conf)

        # check conf
        self.used_method = self.used_config["method"]
        self.fill_classification = self.used_config["fill_classification"]
        self.tile_size = self.used_config["tile_size"]
        self.margin = self.used_config["margin"]
        self.save_intermediate_data = self.used_config["save_intermediate_data"]
        self.fill_nodata = self.used_config["fill_nodata"]

    def check_conf(self, conf):
        """
        Check configuration.
        """

        # init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        overloaded_conf["method"] = conf.get("method", "interpolation")
        overloaded_conf["fill_classification"] = conf.get(
            "fill_classification", "nodata"
        )
        overloaded_conf["fill_nodata"] = conf.get("fill_nodata", None)

        if isinstance(overloaded_conf["fill_classification"], str):
            overloaded_conf["fill_classification"] = [
                overloaded_conf["fill_classification"]
            ]
        overloaded_conf["tile_size"] = conf.get("tile_size", 2000)
        overloaded_conf["margin"] = conf.get("margin", 100)
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        rectification_schema = {
            "method": str,
            "fill_classification": Or(None, [str]),
            "tile_size": int,
            "margin": int,
            "fill_nodata": Or(None, [str]),
            "save_intermediate_data": bool,
        }

        # Check conf
        checker = Checker(rectification_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    @cars_profile(name="Interpolation filling")
    def run(  # pylint: disable=too-many-positional-arguments
        self,
        dsm_file,
        classif_file,
        invalidity_mask_file,
        classif_values,
        dump_dir,
        roi_polys,
        roi_epsg,
        dsm_dir=None,
        orchestrator=None,
    ):
        """
        Run dsm filling by interpolating classified pixels.
        Replaces dsm.tif by the filled dsm. Adds a new band
        to filling.tif if it exists.
        The old dsm is saved in dump_dir.

        roi_poly can any of these objects :
            - a list of Shapely Polygons
            - a Shapely Polygon
        """

        if orchestrator is None:
            orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )

        if dsm_dir is not None:
            dsm_path_out = os.path.join(dsm_dir, "dsm.tif")
        else:
            dsm_path_out = dsm_file

        filling_path_out = os.path.join(dump_dir, "filling.tif")

        if self.fill_classification is None:
            self.fill_classification = ["nodata"]

        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        with rio.open(dsm_file) as in_dsm:
            profile = in_dsm.profile
            profile["crs"] = profile["crs"].to_wkt()
            height = in_dsm.height
            width = in_dsm.width
            dsm_dtype = in_dsm.dtypes[0]
            nodata_value = in_dsm.nodata

        filled_dsm_cars_ds = cars_dataset.CarsDataset(
            "arrays", name="Monoband Filling"
        )
        filled_dsm_cars_ds.create_grid(
            nb_col=width,
            nb_row=height,
            row_split=self.tile_size,
            col_split=self.tile_size,
            row_overlap=self.margin,
            col_overlap=self.margin,
        )

        [saving_info] = orchestrator.get_saving_infos([filled_dsm_cars_ds])

        orchestrator.add_to_save_lists(
            dsm_path_out,
            "interpolation_filled_dsm",
            filled_dsm_cars_ds,
            dtype=dsm_dtype,
            nodata=nodata_value,
            optional_data=False,
            cars_ds_name="interpolation_filled_dsm",
        )

        band_description = [(1, "interpolation")]

        orchestrator.add_to_save_lists(
            filling_path_out,
            "interpolation_filled_filling",
            filled_dsm_cars_ds,
            dtype=np.uint8,
            nodata=0,
            optional_data=False,
            cars_ds_name="interpolation_filled_filling",
            rio_band_description=band_description,
        )

        old_dsm_path = os.path.join(dump_dir, "dsm_not_filled.tif")
        shutil.copy(dsm_file, old_dsm_path)

        if self.save_intermediate_data:
            new_dsm_path = os.path.join(dump_dir, "dsm_filled.tif")
            orchestrator.add_to_save_lists(
                new_dsm_path,
                "interpolation_filled_dsm",
                filled_dsm_cars_ds,
                dtype=dsm_dtype,
                nodata=nodata_value,
                optional_data=False,
                cars_ds_name="interpolation_filled_dsm",
            )

        for row in range(filled_dsm_cars_ds.shape[0]):
            for col in range(filled_dsm_cars_ds.shape[1]):
                full_saving_info = ocht.update_saving_infos(
                    saving_info, row=row, col=col
                )

                window = cars_dataset.window_array_to_dict(
                    filled_dsm_cars_ds.tiling_grid[row, col],
                    filled_dsm_cars_ds.overlaps[row, col],
                )
                overlaps = cars_dataset.overlap_array_to_dict(
                    filled_dsm_cars_ds.overlaps[row, col]
                )
                (
                    filled_dsm_cars_ds[row, col]
                ) = orchestrator.cluster.create_task(
                    interpolation_filling_wrapper, nout=1
                )(
                    old_dsm_path,
                    classif_file,
                    invalidity_mask_file,
                    classif_values,
                    roi_polys,
                    roi_epsg,
                    self.fill_classification,
                    window=window,
                    overlaps=overlaps,
                    fill_nodata=self.fill_nodata,
                    saving_info=full_saving_info,
                    profile=profile,
                )

        return filled_dsm_cars_ds


def interpolation_filling_wrapper(  # pylint: disable=R0917 # noqa: C901
    dsm_file,
    classif_file,
    invalidity_mask_file,
    classif_values,
    roi_polys,
    roi_epsg,
    fill_classification,
    window=None,
    overlaps=None,
    fill_nodata=None,
    saving_info=None,
    profile=None,
):
    """
    Wrapper for interpolation filling, applied on each tile of the DSM.

    :param dsm_file: dsm file to fill
    :param filling_file: filling file
    :param classif_file: classification file
    :return: filled dsm xarray dataset
    """

    col_min = window["col_min"]
    row_min = window["row_min"]
    col_max = window["col_max"]
    row_max = window["row_max"]

    with rio.open(dsm_file) as in_dsm:
        rasterio_window = Window(
            col_off=col_min,
            row_off=row_min,
            width=(col_max - col_min),
            height=(row_max - row_min),
        )

    with rio.open(dsm_file) as in_dsm:
        dsm = in_dsm.read(1, window=rasterio_window).astype(np.float32)
        dsm_mask = in_dsm.read_masks(1, window=rasterio_window)
        dsm_crs = in_dsm.crs
        dsm_nodata = in_dsm.nodata
        window_transform = row_col_to_coords(rasterio_window, in_dsm.transform)

    dsm_valid_mask = dsm_mask != 0
    inside_contour_mask = ndimage.binary_fill_holes(dsm_valid_mask)
    outside_contour_mask = ~inside_contour_mask

    roi_raster = np.ones(dsm.shape, dtype=np.uint8)

    if isinstance(roi_polys, list):
        roi_polys_outepsg = []
        for poly in roi_polys:
            if isinstance(poly, Polygon):
                roi_poly_outepsg = projection.polygon_projection_crs(
                    poly, CRS(roi_epsg), dsm_crs
                )
                roi_polys_outepsg.append(roi_poly_outepsg)

        if roi_polys_outepsg:
            roi_raster = rio.features.rasterize(
                roi_polys_outepsg,
                out_shape=roi_raster.shape,
                transform=window_transform,
            )
    elif isinstance(roi_polys, Polygon):
        roi_poly_outepsg = projection.polygon_projection_crs(
            roi_polys, CRS(roi_epsg), dsm_crs
        )
        roi_raster = rio.features.rasterize(
            [roi_poly_outepsg],
            out_shape=roi_raster.shape,
            transform=window_transform,
        )

    combined_mask = np.zeros_like(dsm, dtype=bool)
    classif = None
    classif_msk = None
    if classif_file is not None:
        with rio.open(classif_file) as in_classif:
            classif = in_classif.read(1, window=rasterio_window)
            classif_msk = in_classif.read_masks(1, window=rasterio_window)

    for label in fill_classification:
        if label in classif_values and classif is not None:
            filling_mask = np.logical_and(classif == int(label), roi_raster > 0)
        elif label == "nodata":
            if classif_msk is not None:
                filling_mask = classif_msk == 0
            else:
                filling_mask = dsm_mask == 0
            filling_mask = np.logical_and(filling_mask, roi_raster > 0)
        else:
            logging.error(
                f"Label {label} not found in classification "
                f"descriptions {classif_values}"
            )
            continue

        logging.info(f"Filling of {label} with rasterio.fill.fillnodata")
        combined_mask = np.logical_or(combined_mask, filling_mask)

    # Keep only targets inside DSM contour to preserve true outside nodata.
    combined_mask = np.logical_and(combined_mask, inside_contour_mask)

    invalidity_mask = None
    if fill_nodata is not None:
        if invalidity_mask_file is not None:
            with rio.open(invalidity_mask_file) as src:
                invalidity_mask = src.read(1, window=rasterio_window)
        for label in fill_nodata:
            filling_mask = np.logical_and(
                invalidity_mask == int(label), roi_raster > 0
            )

            combined_mask = np.logical_or(combined_mask, filling_mask)

    filled_dsm = dsm.copy()
    if np.any(combined_mask) and np.any(
        np.logical_and(dsm_valid_mask, ~combined_mask)
    ):
        fill_value = dsm_nodata if dsm_nodata is not None else 0
        filled_dsm[combined_mask] = fill_value
        filled_dsm = rio.fill.fillnodata(
            filled_dsm,
            mask=np.logical_and(dsm_valid_mask, ~combined_mask),
            max_search_distance=max(dsm.shape),
        )

    if dsm_nodata is not None:
        filled_dsm[outside_contour_mask] = dsm_nodata
    else:
        filled_dsm[outside_contour_mask] = dsm[outside_contour_mask]

    data = {
        "interpolation_filled_dsm": (["row", "col"], filled_dsm),
    }
    coords = {
        "row": np.arange(dsm.shape[0]),
        "col": np.arange(dsm.shape[1]),
    }

    data["interpolation_filled_filling"] = (
        ["row", "col"],
        combined_mask,
    )

    output_dataset = xr.Dataset(
        data_vars=data,
        coords=coords,
    )

    if overlaps is not None:
        core_window = {
            "row_min": window["row_min"] + overlaps["up"],
            "row_max": window["row_max"] - overlaps["down"],
            "col_min": window["col_min"] + overlaps["left"],
            "col_max": window["col_max"] - overlaps["right"],
        }
    else:
        core_window = window

    cars_dataset.fill_dataset(
        output_dataset,
        saving_info=saving_info,
        window=core_window,
        profile=profile,
        attributes=None,
        overlaps=overlaps,
    )

    return output_dataset
