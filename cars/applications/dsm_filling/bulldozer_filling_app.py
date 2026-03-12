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
import warnings

import numpy as np
import rasterio as rio
import xarray as xr
import yaml
from bulldozer.pipeline.bulldozer_pipeline import dsm_to_dtm
from json_checker import Checker, Or
from pyproj import CRS
from rasterio.errors import NodataShadowWarning
from rasterio.windows import Window
from rasterio.windows import transform as row_col_to_coords
from shapely import Polygon

import cars.orchestrator.orchestrator as ocht
from cars.applications.dem_generation.dem_generation_wrappers import (
    edit_transform,
)
from cars.core import inputs, preprocessing, projection, tiling
from cars.data_structures import cars_dataset
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
        orchestrator=None,
        dsm_dir=None,
        tile_size=10000,
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
        if orchestrator is None:
            orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )

        if self.classification is None:
            self.classification = ["nodata"]

        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

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

        # Modify DSM for dtm to be used
        with rio.open(dsm_file) as in_dsm:
            dsm_tr = in_dsm.transform
            dsm_crs = in_dsm.crs
            dsm_bounds = in_dsm.bounds

        saved_transform = None
        if dsm_crs.is_geographic:
            xmin = dsm_bounds.left
            ymin = dsm_bounds.bottom
            utm_epsg = preprocessing.get_utm_zone_as_epsg_code(xmin, ymin)
            conversion_factor = preprocessing.get_conversion_factor(
                dsm_bounds, utm_epsg, dsm_crs.to_epsg()
            )
            resolution = dsm_tr.a * conversion_factor
            saved_transform = edit_transform(dsm_file, resolution=resolution)

        # Launch Bulldozer
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

        # Change back dsm and dtm to previous ref
        if saved_transform is not None:
            edit_transform(dtm_path, transform=saved_transform)
            edit_transform(dsm_file, transform=saved_transform)

        # Generate filled dsm cars dataset

        if dsm_dir is not None:
            dsm_path_out = os.path.join(dsm_dir, "dsm.tif")
            filling_path_out = os.path.join(dsm_dir, "filling.tif")
        else:
            dsm_path_out = dsm_file
            filling_path_out = filling_file

        # get dsm to be filled and its metadata
        with rio.open(dsm_file) as in_dsm:
            profile = in_dsm.profile
            # Transform to wtk for serialization.
            profile["crs"] = profile["crs"].to_wkt()
            height = in_dsm.height
            width = in_dsm.width
            dsm_dtype = in_dsm.dtypes[0]
            nodata_value = in_dsm.nodata

        filled_dsm_cars_ds = cars_dataset.CarsDataset(
            "arrays", name="Monoband Filling"
        )
        # Compute tiling grid
        filled_dsm_cars_ds.tiling_grid = tiling.generate_tiling_grid(
            0,
            0,
            height,
            width,
            tile_size,
            tile_size,
        )

        # Saving infos
        [
            saving_info,
        ] = orchestrator.get_saving_infos([filled_dsm_cars_ds])

        # save list
        orchestrator.add_to_save_lists(
            dsm_path_out,
            "bulldozer_filled_dsm",
            filled_dsm_cars_ds,
            dtype=dsm_dtype,
            nodata=nodata_value,
            optional_data=False,
            cars_ds_name="bulldozer_filled_dsm",
        )

        if filling_file is not None:
            with rio.open(filling_file, "r") as src:
                filling_dtype = src.dtypes[0]
                filling_nodata_value = src.nodata
                # count will be count += 1
                band_description = [
                    (i + 1, src.descriptions[i]) for i in range(src.count)
                ]
                band_description.append(
                    (len(band_description) + 1, "bulldozer")
                )

            orchestrator.add_to_save_lists(
                filling_path_out,
                "bulldozer_filled_filling",
                filled_dsm_cars_ds,
                dtype=filling_dtype,
                nodata=filling_nodata_value,
                optional_data=False,
                cars_ds_name="bulldozer_filled_filling",
                rio_band_description=band_description,
            )

        old_dsm_path = os.path.join(dump_dir, "dsm_not_filled.tif")
        new_dsm_path = os.path.join(dump_dir, "dsm_filled.tif")
        # Save old dsm
        shutil.copy(dsm_file, old_dsm_path)
        if filling_file is not None:
            old_filling_path = os.path.join(dump_dir, "filling_not_filled.tif")
            shutil.copy(filling_file, old_filling_path)

        if self.save_intermediate_data:

            # save new
            orchestrator.add_to_save_lists(
                new_dsm_path,
                "bulldozer_filled_dsm",
                filled_dsm_cars_ds,
                dtype=dsm_dtype,
                nodata=nodata_value,
                optional_data=False,
                cars_ds_name="bulldozer_filled_dsm",
            )

        for row in range(filled_dsm_cars_ds.shape[0]):
            for col in range(filled_dsm_cars_ds.shape[1]):
                # update saving infos  for potential replacement
                full_saving_info = ocht.update_saving_infos(
                    saving_info, row=row, col=col
                )

                window = filled_dsm_cars_ds.get_window_as_dict(row, col)
                # Compute images
                (
                    filled_dsm_cars_ds[row, col]
                ) = orchestrator.cluster.create_task(
                    bulldozer_filling_wrapper, nout=1
                )(
                    old_dsm_path,
                    filling_file,
                    classif_file,
                    dtm_path,
                    roi_polys,
                    roi_epsg,
                    self.classification,
                    window=window,
                    saving_info=full_saving_info,
                    profile=profile,
                )

        return filled_dsm_cars_ds, dtm_path


def bulldozer_filling_wrapper(  # noqa C901 # pylint: disable=R0917
    dsm_file,
    filling_file,
    classif_file,
    dtm_path,
    roi_polys,
    roi_epsg,
    classification,
    window=None,
    saving_info=None,
    profile=None,
):
    """
    Wrapper for exogenous filling, to be applied on each tile of the DSM.

    :param dsm_file:  dsm file to fill
    :param filling_file: filling file
    :param classif_file:  classification file
    :param dtm_path:  dtm file
    :return:  filled dsm xarray dataset
    """

    # Get rasterio window
    col_min = window["col_min"]
    row_min = window["row_min"]
    col_max = window["col_max"]
    row_max = window["row_max"]
    rasterio_window = Window(
        col_off=col_min,
        row_off=row_min,
        width=(col_max - col_min),
        height=(row_max - row_min),
    )

    with rio.open(dsm_file) as in_dsm:
        dsm = in_dsm.read(1, window=rasterio_window)
        dsm_crs = in_dsm.crs

        # Get local transform for window
        window_transform = row_col_to_coords(rasterio_window, in_dsm.transform)

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

    with rio.open(dtm_path) as in_dtm:
        dtm = in_dtm.read(1, window=rasterio_window)

    if classif_file is not None and os.path.exists(classif_file):
        classif_descriptions = inputs.get_descriptions_bands(classif_file)
    else:
        classif_descriptions = []
    combined_mask = np.zeros_like(dsm).astype(np.uint8)
    for label in classification:
        if label in classif_descriptions:
            index_classif = classif_descriptions.index(label) + 1
            with rio.open(classif_file) as in_classif:
                classif = in_classif.read(index_classif, window=rasterio_window)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", NodataShadowWarning)
                    classif_msk = in_classif.read_masks(
                        1, window=rasterio_window
                    )
            classif[classif_msk == 0] = 0

            filling_mask = np.logical_and(classif, roi_raster > 0)
        elif label == "nodata":
            if classif_file is not None and os.path.exists(classif_file):
                with rio.open(classif_file) as in_classif:
                    classif_msk = in_classif.read_masks(
                        1, window=rasterio_window
                    )
                classif = ~classif_msk
            else:
                with rio.open(dsm_file) as in_dsm:
                    dsm_msk = in_dsm.read_masks(1, window=rasterio_window)
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

    data = {
        "bulldozer_filled_dsm": (["row", "col"], dsm),
    }

    coords = {
        "row": np.arange(dsm.shape[0]),
        "col": np.arange(dsm.shape[1]),
    }

    if filling_file is not None:
        # create combined filling
        with rio.open(filling_file) as in_filling:
            nb_bands_filling = in_filling.count + 1
            filling = in_filling.read(window=rasterio_window)
            # add layer combined_mask to filling
            combined_mask = combined_mask[np.newaxis, :, :]
            filling = np.concatenate((filling, combined_mask), axis=0)

        data["bulldozer_filled_filling"] = (
            ["band_filling", "row", "col"],
            filling,
        )
        coords["band_filling"] = np.arange(1, nb_bands_filling + 1)

    output_dataset = xr.Dataset(
        data_vars=data,
        coords=coords,
    )

    cars_dataset.fill_dataset(
        output_dataset,
        saving_info=saving_info,
        window=window,
        profile=profile,
        attributes=None,
        overlaps=None,
    )

    return output_dataset
