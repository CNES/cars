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
import warnings

import numpy as np
import rasterio as rio
import scipy
import skimage
import xarray as xr
from json_checker import Checker, Or
from pyproj import CRS
from rasterio.errors import NodataShadowWarning
from rasterio.windows import Window
from rasterio.windows import transform as row_col_to_coords
from shapely import Polygon

import cars.orchestrator.orchestrator as ocht
from cars.core import inputs, projection, tiling
from cars.data_structures import cars_dataset
from cars.orchestrator.cluster.log_wrapper import cars_profile

from .abstract_dsm_filling_app import DsmFilling


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
        overloaded_conf["method"] = conf.get("method", "border_interpolation")
        overloaded_conf["classification"] = conf.get("classification", "nodata")
        if isinstance(overloaded_conf["classification"], str):
            overloaded_conf["classification"] = [
                overloaded_conf["classification"]
            ]
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

    @cars_profile(name="Border interpolation filling")
    def run(  # pylint: disable=too-many-positional-arguments  # noqa C901
        self,
        dsm_file,
        classif_file,
        filling_file,
        dtm_file,
        dump_dir,
        roi_polys,
        roi_epsg,
        dsm_dir=None,
        orchestrator=None,
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

        if dsm_dir is not None:
            dsm_path_out = os.path.join(dsm_dir, "dsm.tif")
            filling_path_out = os.path.join(dsm_dir, "filling.tif")
        else:
            dsm_path_out = dsm_file
            filling_path_out = filling_file

        if self.classification is None:
            self.classification = ["nodata"]
            logging.error(
                "Filling method 'border_interpolation' needs a classification"
            )

        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

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
            "border_interp_filled_dsm",
            filled_dsm_cars_ds,
            dtype=dsm_dtype,
            nodata=nodata_value,
            optional_data=False,
            cars_ds_name="border_interp_filled_dsm",
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
                    (len(band_description) + 1, "border_interpolation")
                )

            orchestrator.add_to_save_lists(
                filling_path_out,
                "border_interp_filled_filling",
                filled_dsm_cars_ds,
                dtype=filling_dtype,
                nodata=filling_nodata_value,
                optional_data=False,
                cars_ds_name="border_interp_filled_filling",
                rio_band_description=band_description,
            )

        old_dsm_path = os.path.join(dump_dir, "dsm_not_filled.tif")
        new_dsm_path = os.path.join(dump_dir, "dsm_filled.tif")
        old_filling_path = None
        borders_file_path = os.path.join(dump_dir, "borders.tif")
        # Save old dsm
        shutil.copy(dsm_file, old_dsm_path)
        if filling_file is not None:
            old_filling_path = os.path.join(dump_dir, "filling_not_filled.tif")
            shutil.copy(filling_file, old_filling_path)

        if self.save_intermediate_data:
            # save new
            orchestrator.add_to_save_lists(
                new_dsm_path,
                "border_interp_filled_dsm",
                filled_dsm_cars_ds,
                dtype=dsm_dtype,
                nodata=nodata_value,
                optional_data=False,
                cars_ds_name="border_interp_filled_dsm",
            )

            # save borders for labels
            orchestrator.add_to_save_lists(
                borders_file_path,
                "features_boundaries",
                filled_dsm_cars_ds,
                dtype=dsm_dtype,
                nodata=nodata_value,
                optional_data=False,
                cars_ds_name="features_boundaries",
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
                    border_interp_filled_dsm_filling_wrapper, nout=1
                )(
                    old_dsm_path,
                    filling_file,
                    classif_file,
                    dtm_file,
                    roi_polys,
                    roi_epsg,
                    self.classification,
                    self.component_min_size,
                    self.border_size,
                    self.percentile,
                    window=window,
                    saving_info=full_saving_info,
                    profile=profile,
                )

        return filled_dsm_cars_ds


def border_interp_filled_dsm_filling_wrapper(  # noqa C901 # pylint: disable=R0917
    dsm_file,
    filling_file,
    classif_file,
    dtm_file,
    roi_polys,
    roi_epsg,
    classification,
    component_min_size,
    border_size,
    percentile,
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
        dsm_nodata = in_dsm.nodata
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

    # get dtm to fill the dsm
    if dtm_file is not None:
        logging.info(
            "Use DTM file {} for border interpolation".format(dtm_file)
        )
        with rio.open(dtm_file) as in_dtm:
            dtm = in_dtm.read(1, window=rasterio_window)
            dtm_nodata = in_dtm.nodata
    else:
        logging.info(
            "No DTM provided : DSM {} will be used for "
            "border interpolation".format(dsm_file)
        )
        dtm = dsm.copy()
        dtm_nodata = dsm_nodata
    dtm[dtm == dtm_nodata] = np.nan

    if classif_file is not None:
        classif_descriptions = inputs.get_descriptions_bands(classif_file)
    else:
        classif_descriptions = []
    combined_mask = np.zeros_like(dsm).astype(np.uint8)
    stacked_labels = None
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
                (np.ones((component_min_size, 1)), 1),
                (np.ones((1, component_min_size)), 1),
            ],
        )
        features, num_features = scipy.ndimage.label(filling_mask)
        logging.info("Filling of {} features".format(num_features))
        features_boundaries = skimage.morphology.dilation(
            features,
            footprint=[
                (np.ones((border_size, 1)), 1),
                (np.ones((1, border_size)), 1),
            ],
        )
        features_boundaries[filling_mask] = 0

        # concat combined_labels for saving
        if stacked_labels is None:
            stacked_labels = features_boundaries[np.newaxis, :, :]
        else:
            stacked_labels = np.concatenate(
                [stacked_labels, features_boundaries[np.newaxis, :, :]]
            )

        for feature_id in range(1, num_features + 1):
            altitude = np.nanpercentile(
                dtm[features_boundaries == feature_id], percentile
            )
            if altitude is not None:
                dsm[features == feature_id] = altitude
        combined_mask = np.logical_or(combined_mask, filling_mask)

    data = {
        "border_interp_filled_dsm": (["row", "col"], dsm),
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

        data["border_interp_filled_filling"] = (
            ["band_filling", "row", "col"],
            filling,
        )
        coords["band_filling"] = np.arange(1, nb_bands_filling + 1)

    if stacked_labels is not None:
        data["features_boundaries"] = (
            ["nb_labels", "row", "col"],
            stacked_labels,
        )
        coords["nb_labels"] = np.arange(1, stacked_labels.shape[0] + 1)

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
