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
This module contains the exogenous dsm filling application class.
"""

import logging
import os
import shutil
import warnings

import numpy as np
import rasterio as rio
import xarray as xr
from json_checker import Checker, Or
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.errors import NodataShadowWarning
from rasterio.warp import reproject
from rasterio.windows import Window
from rasterio.windows import transform as row_col_to_coords
from shapely import Polygon

import cars.orchestrator.orchestrator as ocht
from cars.core import inputs, projection, tiling
from cars.data_structures import cars_dataset
from cars.orchestrator.cluster.log_wrapper import cars_profile

from .abstract_dsm_filling_app import DsmFilling


class ExogenousFilling(DsmFilling, short_name="exogenous_filling"):
    """
    Exogenous filling
    """

    def __init__(self, conf=None):
        """
        Init function of ExogenousFilling

        :param conf: configuration for ExogenousFilling
        :return: an application_to_use object
        """
        super().__init__(conf=conf)

        # check conf
        self.used_method = self.used_config["method"]
        self.classification = self.used_config["classification"]
        self.fill_with_geoid = self.used_config["fill_with_geoid"]
        self.interpolation_method = self.used_config["interpolation_method"]
        self.save_intermediate_data = self.used_config["save_intermediate_data"]

    def check_conf(self, conf):

        # init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        # Overload conf
        overloaded_conf["method"] = conf.get("method", "exogenous_filling")
        overloaded_conf["classification"] = conf.get("classification", "nodata")
        if isinstance(overloaded_conf["classification"], str):
            overloaded_conf["classification"] = [
                overloaded_conf["classification"]
            ]
        overloaded_conf["fill_with_geoid"] = conf.get("fill_with_geoid", None)
        overloaded_conf["interpolation_method"] = conf.get(
            "interpolation_method", "bilinear"
        )

        if overloaded_conf["interpolation_method"] not in ["bilinear", "cubic"]:
            # pylint: disable=inconsistent-quotes
            raise RuntimeError(
                f"Invalid interpolation method"
                f"{overloaded_conf['interpolation_method']}, "
                f"supported modes are bilinear and cubic."
            )

        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        rectification_schema = {
            "method": str,
            "classification": Or(None, [str]),
            "fill_with_geoid": Or(None, [str]),
            "interpolation_method": str,
            "save_intermediate_data": bool,
        }

        # Check conf
        checker = Checker(rectification_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    @cars_profile(name="Exogeneous filling")
    def run(  # pylint: disable=too-many-positional-arguments  # noqa C901
        self,
        dsm_file,
        classif_file,
        filling_file,
        dump_dir,
        roi_polys,
        roi_epsg,
        output_geoid,
        geom_plugin,
        dsm_dir=None,
        tile_size=10000,
        orchestrator=None,
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

        if self.fill_with_geoid is None:
            self.fill_with_geoid = []

        interpolation_methods_dict = {
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
        }
        interpolation_method = interpolation_methods_dict.get(
            self.interpolation_method, Resampling.bilinear
        )

        if geom_plugin is None:
            logging.error(
                "No DEM was provided, exogenous_filling will not run."
            )
            return None

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
            "exogenous_filled_dsm",
            filled_dsm_cars_ds,
            dtype=dsm_dtype,
            nodata=nodata_value,
            optional_data=False,
            cars_ds_name="exogenous_filled_dsm",
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
                    (len(band_description) + 1, "filling_exogenous")
                )

            orchestrator.add_to_save_lists(
                filling_path_out,
                "exogenous_filled_filling",
                filled_dsm_cars_ds,
                dtype=filling_dtype,
                nodata=filling_nodata_value,
                optional_data=False,
                cars_ds_name="exogenous_filled_filling",
                rio_band_description=band_description,
            )

        old_dsm_path = os.path.join(dump_dir, "dsm_not_filled.tif")
        new_dsm_path = os.path.join(dump_dir, "dsm_filled.tif")
        old_filling_path = None
        # Save old dsm
        shutil.copy(dsm_file, old_dsm_path)
        if filling_file is not None:
            old_filling_path = os.path.join(dump_dir, "filling_not_filled.tif")
            shutil.copy(filling_file, old_filling_path)
        if self.save_intermediate_data:
            # save new
            orchestrator.add_to_save_lists(
                new_dsm_path,
                "exogenous_filled_dsm",
                filled_dsm_cars_ds,
                dtype=dsm_dtype,
                nodata=nodata_value,
                optional_data=False,
                cars_ds_name="exogenous_filled_dsm",
            )

            # saved reprojected dem / intial elevation
            reprojected_dem_path = os.path.join(dump_dir, "reprojected_dem.tif")
            orchestrator.add_to_save_lists(
                reprojected_dem_path,
                "reprojected_dem",
                filled_dsm_cars_ds,
                dtype=dsm_dtype,
                nodata=nodata_value,
                optional_data=False,
                cars_ds_name="reprojected_dem",
            )

            # saved reprojected input geoid
            reprojected_input_geoid_path = os.path.join(
                dump_dir, "reprojected_input_geoid.tif"
            )
            orchestrator.add_to_save_lists(
                reprojected_input_geoid_path,
                "reprojected_input_geoid",
                filled_dsm_cars_ds,
                dtype=dsm_dtype,
                nodata=nodata_value,
                optional_data=False,
                cars_ds_name="reprojected_input_geoid",
            )

            if output_geoid not in (False, None):
                # saved reprojected input geoid
                reprojected_output_geoid_path = os.path.join(
                    dump_dir, "reprojected_output_geoid.tif"
                )
                orchestrator.add_to_save_lists(
                    reprojected_output_geoid_path,
                    "reprojected_output_geoid",
                    filled_dsm_cars_ds,
                    dtype=dsm_dtype,
                    nodata=nodata_value,
                    optional_data=False,
                    cars_ds_name="reprojected_output_geoid",
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
                    exogenous_filling_wrapper, nout=1
                )(
                    old_dsm_path,
                    old_filling_path,
                    classif_file,
                    geom_plugin.dem,
                    geom_plugin.geoid,
                    output_geoid,
                    roi_polys,
                    roi_epsg,
                    self.fill_with_geoid,
                    interpolation_method,
                    self.classification,
                    window=window,
                    saving_info=full_saving_info,
                    profile=profile,
                )

        return filled_dsm_cars_ds


def exogenous_filling_wrapper(  # noqa C901 # pylint: disable=R0917
    dsm_file,
    filling_file,
    classif_file,
    geo_plugin_dem,
    geo_plugin_geoid,
    output_geoid,
    roi_polys,
    roi_epsg,
    fill_with_geoid,
    interpolation_method,
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
    :param geo_plugin_dem: initial elevation dem
    :param geo_plugin_geoid: initial elevation geoid
    :param output_geoid:  output geoid
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

    # Get the initial elevation
    with rio.open(geo_plugin_dem) as in_elev:
        # Reproject the elevation data to match the DSM
        elev_data = np.empty(dsm.shape, dtype=in_elev.dtypes[0])

        reproject(
            source=rio.band(in_elev, 1),
            destination=elev_data,
            src_transform=in_elev.transform,
            src_crs=in_elev.crs,
            dst_transform=window_transform,
            dst_crs=dsm_crs,
            resampling=interpolation_method,
        )

    with rio.open(geo_plugin_geoid) as in_geoid:
        # Reproject the geoid data to match the DSM
        input_geoid_data = np.empty(dsm.shape, dtype=in_geoid.dtypes[0])

        reproject(
            source=rio.band(in_geoid, 1),
            destination=input_geoid_data,
            src_transform=in_geoid.transform,
            src_crs=in_geoid.crs,
            dst_transform=window_transform,
            dst_crs=dsm_crs,
            resampling=interpolation_method,
        )

    output_geoid_data = None
    if isinstance(output_geoid, str):
        with rio.open(output_geoid) as in_geoid:
            # Reproject the geoid data to match the DSM
            output_geoid_data = np.empty(dsm.shape, dtype=in_geoid.dtypes[0])

            reproject(
                source=rio.band(in_geoid, 1),
                destination=output_geoid_data,
                src_transform=in_geoid.transform,
                src_crs=in_geoid.crs,
                dst_transform=window_transform,
                dst_crs=dsm_crs,
                resampling=interpolation_method,
            )

    # Fill DSM for every label
    combined_mask = np.zeros_like(dsm).astype(np.uint8)
    if classif_file is not None:
        classif_descriptions = inputs.get_descriptions_bands(classif_file)
    else:
        classif_descriptions = []
    for label in classification:
        if label in classif_descriptions:
            index_classif = classif_descriptions.index(label) + 1
            with rio.open(classif_file) as in_classif:
                classif = in_classif.read(index_classif, window=rasterio_window)
                classif_msk = in_classif.read_masks(1, window=rasterio_window)
            classif[classif_msk == 0] = 0
            filling_mask = np.logical_and(classif, roi_raster > 0)
        elif label == "nodata":
            if classif_file is not None:
                with rio.open(classif_file) as in_classif:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", NodataShadowWarning)
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

        if label in fill_with_geoid:
            logging.info("Filling of {} with geoid".format(label))
            dsm[filling_mask] = 0
        else:
            logging.info("Filling of {} with DEM and geoid".format(label))
            dsm[filling_mask] = elev_data[filling_mask]

        # apply offset to project on geoid if needed
        if output_geoid is not True:
            if isinstance(output_geoid, bool) and output_geoid is False:
                # out geoid is ellipsoid: add geoid-ellipsoid distance
                dsm[filling_mask] += input_geoid_data[filling_mask]
            elif isinstance(output_geoid, str):
                # out geoid is a new geoid whose path is in output_geoid:
                # add carsgeoid-ellipsoid then add ellipsoid-outgeoid
                dsm[filling_mask] += input_geoid_data[filling_mask]
                dsm[filling_mask] -= output_geoid_data[filling_mask]

        combined_mask = np.logical_or(combined_mask, filling_mask)

    data = {
        "exogenous_filled_dsm": (["row", "col"], dsm),
        "reprojected_dem": (["row", "col"], elev_data),
        "reprojected_input_geoid": (["row", "col"], input_geoid_data),
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

        data["exogenous_filled_filling"] = (
            ["band_filling", "row", "col"],
            filling,
        )
        coords["band_filling"] = np.arange(1, nb_bands_filling + 1)

    if output_geoid_data is not None:
        data["reprojected_output_geoid"] = (["row", "col"], output_geoid_data)

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
