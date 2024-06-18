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
# pylint: disable=too-many-lines

"""
this module contains the dense_matching application class.
"""

import collections

# Third party imports
import copy

# Standard imports
import logging
import os
from typing import List

import numpy as np
import rasterio as rio
import xarray
from affine import Affine
from json_checker import Checker, Or

# CARS imports
import cars.applications.rasterization.rasterization_tools as rasterization_step
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.point_cloud_fusion import point_cloud_tools
from cars.applications.rasterization import (
    rasterization_constants as raster_cst,
)
from cars.applications.rasterization.point_cloud_rasterization import (
    PointCloudRasterization,
)
from cars.core import constants as cst
from cars.core import projection, tiling
from cars.data_structures import cars_dataset, format_transformation

# R0903  temporary disabled for error "Too few public methods"
# œgoing to be corrected by adding new methods as check_conf


class SimpleGaussian(
    PointCloudRasterization, short_name="simple_gaussian"
):  # pylint: disable=R0903
    """
    PointsCloudRasterisation
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf=None):
        """
        Init function of PointsCloudRasterisation

        :param conf: configuration for rasterization
        :return: a application_to_use object
        """

        super().__init__(conf=conf)

        # check conf

        # get rasterization parameter
        self.used_method = self.used_config["method"]
        self.dsm_radius = self.used_config["dsm_radius"]
        self.sigma = self.used_config["sigma"]
        self.grid_points_division_factor = self.used_config[
            "grid_points_division_factor"
        ]
        self.resolution = self.used_config["resolution"]
        # get nodata values
        self.dsm_no_data = self.used_config["dsm_no_data"]
        self.color_no_data = self.used_config["color_no_data"]
        self.color_dtype = self.used_config["color_dtype"]
        self.msk_no_data = self.used_config["msk_no_data"]
        # Get if color, mask and stats are saved
        self.save_color = self.used_config["save_color"]
        self.save_stats = self.used_config["save_stats"]
        self.save_mask = self.used_config["save_mask"]
        self.save_classif = self.used_config["save_classif"]
        self.save_dsm = self.used_config["save_dsm"]
        self.save_intervals = self.used_config["save_intervals"]
        self.save_confidence = self.used_config["save_confidence"]
        self.save_source_pc = self.used_config["save_source_pc"]
        self.save_filling = self.used_config["save_filling"]

        # Init orchestrator
        self.orchestrator = None

    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check
        :type conf: dict

        :return: overloaded configuration
        :rtype: dict

        """

        # init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        # Overload conf

        # get rasterization parameter
        overloaded_conf["method"] = conf.get("method", "simple_gaussian")
        overloaded_conf["dsm_radius"] = conf.get("dsm_radius", 1)
        overloaded_conf["sigma"] = conf.get("sigma", None)
        overloaded_conf["grid_points_division_factor"] = conf.get(
            "grid_points_division_factor", None
        )
        overloaded_conf["resolution"] = conf.get("resolution", 0.5)

        # get nodata values
        overloaded_conf["dsm_no_data"] = conf.get("dsm_no_data", -32768)
        overloaded_conf["color_no_data"] = conf.get("color_no_data", 0)
        overloaded_conf["color_dtype"] = conf.get("color_dtype", None)
        overloaded_conf["msk_no_data"] = conf.get("msk_no_data", 255)

        # Get if color, mask and stats are saved
        overloaded_conf["save_color"] = conf.get("save_color", True)
        overloaded_conf["save_stats"] = conf.get("save_stats", False)
        overloaded_conf["save_mask"] = conf.get("save_mask", False)
        overloaded_conf["save_classif"] = conf.get("save_classif", False)
        overloaded_conf["save_dsm"] = conf.get("save_dsm", True)
        overloaded_conf["save_intervals"] = conf.get("save_intervals", False)
        overloaded_conf["save_confidence"] = conf.get("save_confidence", False)
        overloaded_conf["save_source_pc"] = conf.get("save_source_pc", False)
        overloaded_conf["save_filling"] = conf.get("save_filling", False)

        overloaded_conf["compute_all"] = conf.get("compute_all", False)
        if overloaded_conf["compute_all"]:
            # all the layers will computed
            self.list_computed_layers = None
        else:
            # only the saved layers will be saved
            self.list_computed_layers = []
            for key in overloaded_conf.keys():
                if "save_" in key and overloaded_conf[key]:
                    self.list_computed_layers.append(key.split("save_")[1])

        rasterization_schema = {
            "method": str,
            "resolution": float,
            "dsm_radius": Or(float, int),
            "sigma": Or(float, None),
            "grid_points_division_factor": Or(None, int),
            "dsm_no_data": int,
            "msk_no_data": int,
            "color_no_data": int,
            "color_dtype": Or(None, str),
            "save_color": bool,
            "save_mask": bool,
            "save_classif": bool,
            "save_stats": bool,
            "save_dsm": bool,
            "save_intervals": bool,
            "save_confidence": bool,
            "save_source_pc": bool,
            "save_filling": bool,
            "compute_all": bool,
        }

        # Check conf
        checker = Checker(rasterization_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def get_resolution(self):
        """
        Get the resolution used by rasterization application

        :return: resolution in meters or degrees

        """

        return self.resolution

    def get_margins(self):
        """
        Get the margin to use for terrain tiles

        :return: margin in meters or degrees
        """

        margins = self.dsm_radius * self.resolution
        return margins

    def get_optimal_tile_size(
        self,
        max_ram_per_worker,
        superposing_point_clouds=1,
        point_cloud_resolution=0.5,
    ):
        """
        Get the optimal tile size to use, depending on memory available

        :param max_ram_per_worker: maximum ram available
        :type max_ram_per_worker: int
        :param superposing_point_clouds: number of point clouds superposing
        :type superposing_point_clouds: int
        :param point_cloud_resolution: resolution of point cloud
        :type point_cloud_resolution: float

        :return: optimal tile size in meter
        :rtype: float

        """

        tot = 7000 * superposing_point_clouds / point_cloud_resolution

        import_ = 200  # MiB
        tile_size = int(
            np.sqrt(float(((max_ram_per_worker - import_) * 2**23)) / tot)
        )

        logging.info(
            "Estimated optimal tile size for rasterization: {} meters".format(
                tile_size
            )
        )
        return tile_size

    def run(  # noqa: C901 function is too complex
        self,
        points_clouds,
        epsg,
        orchestrator=None,
        dsm_file_name=None,
        color_file_name=None,
        color_dtype=None,
    ):
        """
        Run PointsCloudRasterisation application.

        Creates a CarsDataset filled with dsm tiles.

        :param points_clouds: merged point cloud or list of array points clouds

            . CarsDataset contains:

            - Z x W Delayed tiles. \
                Each tile will be a future pandas DataFrame containing:

                - data with keys  "x", "y", "z", "corr_msk" \
                    optional: "color", "mask", "data_valid", "z_inf", "z_sup"\
                      "coord_epi_geom_i", "coord_epi_geom_j", "idx_im_epi"
                - attrs with keys "epsg", "ysize", "xsize", "xstart", "ystart"

             - attributes containing "bounds", "ysize", "xsize", "epsg"

             OR


             Tuple(list of CarsDataset Arrays, bounds). With list of point
                clouds:
                list of CarsDataset of type array, with:
                - data with keys x", "y", "z", "corr_msk", "z_inf", "z_sup"\
                    optional: "color", "mask", "data_valid",\
                      "coord_epi_geom_i", "coord_epi_geom_j", "idx_im_epi"

        :type points_clouds: CarsDataset filled with pandas.DataFrame
        :param epsg: epsg of raster data
        :type epsg: str
        :param orchestrator: orchestrator used
        :param dsm_file_name: path of dsm
        :type dsm_file_name: str
        :param color_file_name: path of color
        :type color_file_name: str
        :param color_dtype: output color image type
        :type color_dtype: str (numpy type)

        :return: raster DSM. CarsDataset contains:

            - Z x W Delayed tiles. \
                Each tile will be a future xarray Dataset containing:

                - data : with keys : "hgt", "img", "raster_msk",optional : \
                  "n_pts", "pts_in_cell", "hgt_mean", "hgt_stdev",\
                  "hgt_inf", "hgt_sup"
                - attrs with keys: "epsg"
            - attributes containing: None

        :rtype : CarsDataset filled with xr.Dataset
        """

        # Default orchestrator
        if orchestrator is None:
            # Create default sequential orchestrator for current application
            # be awere, no out_json will be shared between orchestrators
            # No files saved
            self.orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )
        else:
            self.orchestrator = orchestrator

        # Check if input data is supported
        data_valid = True
        if isinstance(points_clouds, tuple):
            if isinstance(points_clouds[0][0], cars_dataset.CarsDataset):
                if points_clouds[0][0].dataset_type not in ("arrays", "points"):
                    data_valid = False
            else:
                data_valid = False
        elif isinstance(points_clouds, cars_dataset.CarsDataset):
            if points_clouds.dataset_type != "points":
                data_valid = False
        else:
            data_valid = False
        if not data_valid:
            message = (
                "PointsCloudRasterisation application doesn't support "
                "this input data "
                "format : type : {}".format(type(points_clouds))
            )
            logging.error(message)
            raise RuntimeError(message)

        # Create CarsDataset
        terrain_raster = cars_dataset.CarsDataset(
            "arrays", name="rasterization"
        )

        if isinstance(points_clouds, cars_dataset.CarsDataset):
            # Get tiling grid
            terrain_raster.tiling_grid = (
                format_transformation.terrain_coords_to_pix(
                    points_clouds, self.resolution
                )
            )
            bounds = points_clouds.attributes["bounds"]
        else:
            bounds = points_clouds[1]
            # tiling grid: all tiles from sources -> not replaceable.
            # CarsDataset is only used for processing
            nb_tiles = 0
            for point_cld in points_clouds[0]:
                nb_tiles += point_cld.shape[0] * point_cld.shape[1]
            terrain_raster.tiling_grid = tiling.generate_tiling_grid(
                0, 0, 1, nb_tiles, 1, 1
            )

        terrain_raster.generate_none_tiles()

        # Derive output image files parameters to pass to rasterio
        xsize, ysize = tiling.roi_to_start_and_size(bounds, self.resolution)[2:]
        logging.info("DSM output image size: {}x{} pixels".format(xsize, ysize))

        if isinstance(points_clouds, tuple):
            source_pc_names = points_clouds[0][0].attributes["source_pc_names"]
        else:
            source_pc_names = points_clouds.attributes["source_pc_names"]

        # Save objects
        # Initialize files names
        # TODO get from config ?
        out_dsm_file_name = None
        out_dsm_inf_file_name = None
        out_dsm_sup_file_name = None
        out_weights_file_name = None
        out_clr_file_name = None
        out_msk_file_name = None
        out_confidence = None
        out_dsm_mean_file_name = None
        out_dsm_std_file_name = None
        out_dsm_n_pts_file_name = None
        out_dsm_points_in_cell_file_name = None
        out_dsm_inf_mean_file_name = None
        out_dsm_inf_std_file_name = None
        out_dsm_inf_n_pts_file_name = None
        out_dsm_inf_points_in_cell_file_name = None
        out_dsm_sup_mean_file_name = None
        out_dsm_sup_std_file_name = None
        out_dsm_sup_n_pts_file_name = None
        out_dsm_sup_points_in_cell_file_name = None

        if self.save_dsm:
            if dsm_file_name is not None:
                out_dsm_file_name = dsm_file_name
            else:
                out_dsm_file_name = os.path.join(
                    self.orchestrator.out_dir, "dsm.tif"
                )
            self.orchestrator.add_to_save_lists(
                out_dsm_file_name,
                cst.RASTER_HGT,
                terrain_raster,
                dtype=np.float32,
                nodata=self.dsm_no_data,
                cars_ds_name="dsm",
            )
            out_weights_file_name = os.path.join(
                self.orchestrator.out_dir, "weights.tif"
            )
            self.orchestrator.add_to_save_lists(
                out_weights_file_name,
                cst.RASTER_WEIGHTS_SUM,
                terrain_raster,
                dtype=np.float32,
                nodata=0,
                cars_ds_name="dsm_weights",
            )

        # TODO Check that intervals indeed exist!
        if self.save_intervals:
            out_dsm_inf_file_name = os.path.join(
                self.orchestrator.out_dir, "dsm_inf.tif"
            )
            self.orchestrator.add_to_save_lists(
                out_dsm_inf_file_name,
                cst.RASTER_HGT_INF,
                terrain_raster,
                dtype=np.float32,
                nodata=self.dsm_no_data,
                cars_ds_name="dsm_inf",
            )
            out_dsm_sup_file_name = os.path.join(
                self.orchestrator.out_dir, "dsm_sup.tif"
            )
            self.orchestrator.add_to_save_lists(
                out_dsm_sup_file_name,
                cst.RASTER_HGT_SUP,
                terrain_raster,
                dtype=np.float32,
                nodata=self.dsm_no_data,
                cars_ds_name="dsm_sup",
            )

        if self.save_color:
            if color_file_name is not None:
                out_clr_file_name = color_file_name
            else:
                out_clr_file_name = os.path.join(
                    self.orchestrator.out_dir, "clr.tif"
                )
            if not self.color_dtype:
                self.color_dtype = color_dtype
            self.orchestrator.add_to_save_lists(
                out_clr_file_name,
                cst.RASTER_COLOR_IMG,
                terrain_raster,
                dtype=self.color_dtype,
                nodata=self.color_no_data,
                cars_ds_name="color",
            )
        if self.save_stats:
            out_dsm_mean_file_name = os.path.join(
                self.orchestrator.out_dir, "dsm_mean.tif"
            )
            out_dsm_std_file_name = os.path.join(
                self.orchestrator.out_dir, "dsm_std.tif"
            )
            out_dsm_n_pts_file_name = os.path.join(
                self.orchestrator.out_dir, "dsm_n_pts.tif"
            )
            out_dsm_points_in_cell_file_name = os.path.join(
                self.orchestrator.out_dir, "dsm_pts_in_cell.tif"
            )
            self.orchestrator.add_to_save_lists(
                out_dsm_mean_file_name,
                cst.RASTER_HGT_MEAN,
                terrain_raster,
                dtype=np.float32,
                nodata=self.dsm_no_data,
                cars_ds_name="dsm_mean",
            )
            self.orchestrator.add_to_save_lists(
                out_dsm_std_file_name,
                cst.RASTER_HGT_STD_DEV,
                terrain_raster,
                dtype=np.float32,
                nodata=self.dsm_no_data,
                cars_ds_name="dsm_std",
            )
            self.orchestrator.add_to_save_lists(
                out_dsm_n_pts_file_name,
                cst.RASTER_NB_PTS,
                terrain_raster,
                dtype=np.uint16,
                nodata=0,
                cars_ds_name="dsm_n_pts",
            )
            self.orchestrator.add_to_save_lists(
                out_dsm_points_in_cell_file_name,
                cst.RASTER_NB_PTS_IN_CELL,
                terrain_raster,
                dtype=np.uint16,
                nodata=0,
                cars_ds_name="dsm_pts_in_cells",
            )
            if self.save_intervals:
                out_dsm_inf_mean_file_name = os.path.join(
                    self.orchestrator.out_dir, "dsm_inf_mean.tif"
                )
                out_dsm_inf_std_file_name = os.path.join(
                    self.orchestrator.out_dir, "dsm_inf_std.tif"
                )
                self.orchestrator.add_to_save_lists(
                    out_dsm_inf_mean_file_name,
                    cst.RASTER_HGT_INF_MEAN,
                    terrain_raster,
                    dtype=np.float32,
                    nodata=self.dsm_no_data,
                    cars_ds_name="dsm_inf_mean",
                )
                self.orchestrator.add_to_save_lists(
                    out_dsm_inf_std_file_name,
                    cst.RASTER_HGT_INF_STD_DEV,
                    terrain_raster,
                    dtype=np.float32,
                    nodata=self.dsm_no_data,
                    cars_ds_name="dsm_inf_std",
                )
                out_dsm_sup_mean_file_name = os.path.join(
                    self.orchestrator.out_dir, "dsm_sup_mean.tif"
                )
                out_dsm_sup_std_file_name = os.path.join(
                    self.orchestrator.out_dir, "dsm_sup_std.tif"
                )
                self.orchestrator.add_to_save_lists(
                    out_dsm_sup_mean_file_name,
                    cst.RASTER_HGT_SUP_MEAN,
                    terrain_raster,
                    dtype=np.float32,
                    nodata=self.dsm_no_data,
                    cars_ds_name="dsm_sup_mean",
                )
                self.orchestrator.add_to_save_lists(
                    out_dsm_sup_std_file_name,
                    cst.RASTER_HGT_SUP_STD_DEV,
                    terrain_raster,
                    dtype=np.float32,
                    nodata=self.dsm_no_data,
                    cars_ds_name="dsm_sup_std",
                )
        if self.save_classif:
            out_classif_file_name = os.path.join(
                self.orchestrator.out_dir, "classif.tif"
            )
            self.orchestrator.add_to_save_lists(
                out_classif_file_name,
                cst.RASTER_CLASSIF,
                terrain_raster,
                dtype=np.uint8,
                nodata=self.msk_no_data,
                cars_ds_name="dsm_classif",
            )
        if self.save_mask:
            out_msk_file_name = os.path.join(
                self.orchestrator.out_dir, "msk.tif"
            )
            self.orchestrator.add_to_save_lists(
                out_msk_file_name,
                cst.RASTER_MSK,
                terrain_raster,
                dtype=np.uint8,
                nodata=self.msk_no_data,
                cars_ds_name="dsm_mask",
            )

        if self.save_confidence:
            out_confidence = os.path.join(
                self.orchestrator.out_dir, "confidence.tif"
            )
            self.orchestrator.add_to_save_lists(
                out_confidence,
                cst.RASTER_CONFIDENCE,
                terrain_raster,
                dtype=np.float32,
                nodata=self.msk_no_data,
                cars_ds_name="confidence",
            )

        if self.save_source_pc:
            out_source_pc = os.path.join(
                self.orchestrator.out_dir, "source_pc.tif"
            )
            self.orchestrator.add_to_save_lists(
                out_source_pc,
                cst.RASTER_SOURCE_PC,
                terrain_raster,
                dtype=np.uint8,
                nodata=self.msk_no_data,
                cars_ds_name="source_pc",
            )

        if self.save_filling:
            out_filling = os.path.join(self.orchestrator.out_dir, "filling.tif")
            self.orchestrator.add_to_save_lists(
                out_filling,
                cst.RASTER_FILLING,
                terrain_raster,
                dtype=np.uint8,
                nodata=self.msk_no_data,
                cars_ds_name="filling",
            )

        # Get saving infos in order to save tiles when they are computed
        [saving_info] = self.orchestrator.get_saving_infos([terrain_raster])

        # Generate profile
        geotransform = (
            bounds[0],
            self.resolution,
            0.0,
            bounds[3],
            0.0,
            -self.resolution,
        )

        transform = Affine.from_gdal(*geotransform)
        raster_profile = collections.OrderedDict(
            {
                "height": ysize,
                "width": xsize,
                "driver": "GTiff",
                "dtype": "float32",
                "transform": transform,
                "crs": "EPSG:{}".format(epsg),
                "tiled": True,
            }
        )

        # Get number of tiles
        logging.info(
            "Number of tiles in cloud rasterization: "
            "row: {} "
            "col: {}".format(
                terrain_raster.tiling_grid.shape[0],
                terrain_raster.tiling_grid.shape[1],
            )
        )

        # Add infos to orchestrator.out_json
        updating_dict = {
            application_constants.APPLICATION_TAG: {
                raster_cst.RASTERIZATION_RUN_TAG: {
                    raster_cst.EPSG_TAG: epsg,
                    raster_cst.DSM_TAG: out_dsm_file_name,
                    raster_cst.DSM_INF_TAG: out_dsm_inf_file_name,
                    raster_cst.DSM_SUP_TAG: out_dsm_sup_file_name,
                    raster_cst.DSM_NO_DATA_TAG: float(self.dsm_no_data),
                    raster_cst.COLOR_NO_DATA_TAG: float(self.color_no_data),
                    raster_cst.COLOR_TAG: out_clr_file_name,
                    raster_cst.MSK_TAG: out_msk_file_name,
                    raster_cst.CONFIDENCE_TAG: out_confidence,
                    raster_cst.DSM_MEAN_TAG: out_dsm_mean_file_name,
                    raster_cst.DSM_STD_TAG: out_dsm_std_file_name,
                    raster_cst.DSM_N_PTS_TAG: out_dsm_n_pts_file_name,
                    raster_cst.DSM_POINTS_IN_CELL_TAG: (
                        out_dsm_points_in_cell_file_name
                    ),
                    raster_cst.DSM_INF_MEAN_TAG: out_dsm_inf_mean_file_name,
                    raster_cst.DSM_INF_STD_TAG: out_dsm_inf_std_file_name,
                    raster_cst.DSM_INF_N_PTS_TAG: out_dsm_inf_n_pts_file_name,
                    raster_cst.DSM_INF_POINTS_IN_CELL_TAG: (
                        out_dsm_inf_points_in_cell_file_name
                    ),
                    raster_cst.DSM_SUP_MEAN_TAG: out_dsm_sup_mean_file_name,
                    raster_cst.DSM_SUP_STD_TAG: out_dsm_sup_std_file_name,
                    raster_cst.DSM_SUP_N_PTS_TAG: out_dsm_sup_n_pts_file_name,
                    raster_cst.DSM_SUP_POINTS_IN_CELL_TAG: (
                        out_dsm_sup_points_in_cell_file_name
                    ),
                },
            }
        }
        self.orchestrator.update_out_info(updating_dict)

        # Generate rasters
        if not isinstance(points_clouds, tuple):
            for col in range(terrain_raster.shape[1]):
                for row in range(terrain_raster.shape[0]):
                    # get corresponding point cloud
                    # one tile in point cloud correspond
                    # to one tile in raster
                    # uses rasterio conventions
                    (
                        pc_row,
                        pc_col,
                    ) = format_transformation.get_corresponding_indexes(
                        row, col
                    )

                    if points_clouds.tiles[pc_row][pc_col] is not None:
                        # Get window
                        window = cars_dataset.window_array_to_dict(
                            terrain_raster.tiling_grid[row, col]
                        )
                        # update saving infos  for potential replacement
                        full_saving_info = ocht.update_saving_infos(
                            saving_info, row=row, col=col
                        )

                        # Get terrain region
                        # corresponding to point cloud tile
                        terrain_region = [
                            points_clouds.tiling_grid[pc_row, pc_col, 0],
                            points_clouds.tiling_grid[pc_row, pc_col, 2],
                            points_clouds.tiling_grid[pc_row, pc_col, 1],
                            points_clouds.tiling_grid[pc_row, pc_col, 3],
                        ]

                        # Delayed call to rasterization operations using all
                        #  required point clouds
                        terrain_raster[
                            row, col
                        ] = self.orchestrator.cluster.create_task(
                            rasterization_wrapper
                        )(
                            points_clouds[pc_row, pc_col],
                            self.resolution,
                            epsg,
                            raster_profile,
                            window=window,
                            terrain_region=terrain_region,
                            list_computed_layers=self.list_computed_layers,
                            saving_info=full_saving_info,
                            radius=self.dsm_radius,
                            sigma=self.sigma,
                            dsm_no_data=self.dsm_no_data,
                            color_no_data=self.color_no_data,
                            msk_no_data=self.msk_no_data,
                            source_pc_names=source_pc_names,
                        )
        else:

            # Add final function to apply
            terrain_raster.final_function = raster_final_function
            ind_tile = 0
            for point_cloud in points_clouds[0]:
                for row_pc in range(point_cloud.shape[0]):
                    for col_pc in range(point_cloud.shape[1]):
                        # update saving infos  for potential replacement
                        full_saving_info = ocht.update_saving_infos(
                            saving_info, row=0, col=ind_tile
                        )
                        if point_cloud[row_pc, col_pc] is not None:
                            # Delayed call to rasterization operations using all
                            #  required point clouds
                            terrain_raster[
                                0, ind_tile
                            ] = self.orchestrator.cluster.create_task(
                                rasterization_wrapper
                            )(
                                point_cloud[row_pc, col_pc],
                                self.resolution,
                                epsg,
                                raster_profile,
                                window=None,
                                terrain_region=None,
                                terrain_full_roi=bounds,
                                list_computed_layers=self.list_computed_layers,
                                saving_info=full_saving_info,
                                radius=self.dsm_radius,
                                sigma=self.sigma,
                                dsm_no_data=self.dsm_no_data,
                                color_no_data=self.color_no_data,
                                msk_no_data=self.msk_no_data,
                                source_pc_names=source_pc_names,
                            )
                        ind_tile += 1

        # Sort tiles according to rank TODO remove or implement it ?

        return terrain_raster


def rasterization_wrapper(
    cloud,
    resolution,
    epsg,
    profile,
    window=None,
    terrain_region=None,
    terrain_full_roi=None,
    list_computed_layers: List[str] = None,
    saving_info=None,
    sigma: float = None,
    radius: int = 1,
    dsm_no_data: int = np.nan,
    color_no_data: int = np.nan,
    msk_no_data: int = 255,
    source_pc_names=None,
):
    """
    Wrapper for rasterization step :
    - Convert a list of clouds to correct epsg
    - Rasterize it with associated colors

    if terrain_region is not provided: region is computed from point cloud,
        with margin to use

    :param cloud: combined cloud
    :type cloud: pandas.DataFrame
    :param terrain_region: terrain bounds
    :param resolution: Produced DSM resolution (meter, degree [EPSG dependent])
    :type resolution: float
    :param  epsg_code: epsg code for the CRS of the output DSM
    :type epsg_code: int
    :param  window: Window considered
    :type window: int
    :param  margin: margin in pixel to use
    :type margin: int
    :param  profile: rasterio profile
    :param list_computed_layers: list of computed output data
    :type profile: dict
    :param saving_info: information about CarsDataset ID.
    :type saving_info: dict
    :param sigma: sigma for gaussian interpolation.
        (If None, set to resolution)
    :param radius: Radius for hole filling.
    :param dsm_no_data: no data value to use in the final raster
    :param color_no_data: no data value to use in the final colored raster
    :param msk_no_data: no data value to use in the final mask image
    :param source_pc_names: list of names of points cloud before merging :
        name of sensors pair or name of point cloud file
    :return: digital surface model + projected colors
    :rtype: xr.Dataset
    """
    # update attributes
    attributes = copy.deepcopy(cloud.attrs)
    attributes.update(attributes.get("attributes", {}))
    if "attributes" in attributes:
        del attributes["attributes"]
    if "saving_info" in attributes:
        del attributes["saving_info"]

    # convert back to correct epsg
    # If the points cloud is not in the right epsg referential, it is converted
    if isinstance(cloud, xarray.Dataset):
        # Transform Dataset to Dataframe
        cloud, cloud_epsg = point_cloud_tools.create_combined_cloud(
            [cloud], [attributes["cloud_id"]], epsg
        )
    elif cloud is None:
        logging.warning("Input cloud is None")
        return None
    else:
        cloud_epsg = attributes.get("epsg")

    if "number_of_pc" not in attributes and source_pc_names is not None:
        attributes["number_of_pc"] = len(source_pc_names)

    # update attributes
    cloud.attrs = {}
    cars_dataset.fill_dataframe(cloud, attributes=attributes)

    if epsg != cloud_epsg:
        projection.points_cloud_conversion_dataframe(cloud, cloud_epsg, epsg)

    # filter cloud
    if "mask" in cloud:
        cloud = cloud[cloud["mask"] == 0]

    if cloud.dropna().empty:
        return None

    # Compute start and size
    if terrain_region is None:
        # compute region from cloud
        xmin = np.nanmin(cloud["x"])
        xmax = np.nanmax(cloud["x"])
        ymin = np.nanmin(cloud["y"])
        ymax = np.nanmax(cloud["y"])
        # Add margin to be sure every point is rasterized
        terrain_region = [
            xmin - radius * resolution,
            ymin - radius * resolution,
            xmax + radius * resolution,
            ymax + radius * resolution,
        ]

        if terrain_full_roi is not None:
            # Modify start (used in tiling.roi_to_start_and_size)  [0, 3]
            # to share the same global grid
            terrain_region[0] = (
                terrain_full_roi[0]
                + np.round(
                    (terrain_region[0] - terrain_full_roi[0]) / resolution
                )
                * resolution
            )
            terrain_region[3] = (
                terrain_full_roi[3]
                + np.round(
                    (terrain_region[3] - terrain_full_roi[3]) / resolution
                )
                * resolution
            )
            # Crop
            terrain_region = [
                max(terrain_full_roi[0], terrain_region[0]),
                max(terrain_full_roi[1], terrain_region[1]),
                min(terrain_full_roi[2], terrain_region[2]),
                min(terrain_full_roi[3], terrain_region[3]),
            ]

            if (
                terrain_region[0] > terrain_region[2]
                or terrain_region[1] > terrain_region[3]
            ):
                return None

    xstart, ystart, xsize, ysize = tiling.roi_to_start_and_size(
        terrain_region, resolution
    )
    if window is None:
        transform = rio.Affine(*profile["transform"][0:6])
        row_pix_pos, col_pix_pos = rio.transform.AffineTransformer(
            transform
        ).rowcol(xstart, ystart)
        window = [
            row_pix_pos,
            row_pix_pos + ysize,
            col_pix_pos,
            col_pix_pos + xsize,
        ]

        window = cars_dataset.window_array_to_dict(window)

    # Call simple_rasterization
    raster = rasterization_step.simple_rasterization_dataset_wrapper(
        cloud,
        resolution,
        epsg,
        xstart=xstart,
        ystart=ystart,
        xsize=xsize,
        ysize=ysize,
        sigma=sigma,
        radius=radius,
        dsm_no_data=dsm_no_data,
        color_no_data=color_no_data,
        msk_no_data=msk_no_data,
        list_computed_layers=list_computed_layers,
        source_pc_names=source_pc_names,
    )

    # Fill raster
    if raster is not None:
        cars_dataset.fill_dataset(
            raster,
            saving_info=saving_info,
            window=window,
            profile=profile,
            attributes=None,
            overlaps=None,
        )

    return raster


def raster_final_function(orchestrator, future_object):
    """
    Apply function to current object, reading already rasterized data

    :param orchestrator: orchestrator
    :param future_object: Dataset

    :return: update object
    """
    # Get data weights
    old_weights, _ = orchestrator.get_data(
        cst.RASTER_WEIGHTS_SUM, future_object
    )
    weights = future_object[cst.RASTER_WEIGHTS_SUM].values

    future_object[cst.RASTER_WEIGHTS_SUM].values = np.reshape(
        rasterization_step.update_weights(old_weights, weights), weights.shape
    )

    # Get data dsm

    for tag in future_object.keys():

        if tag != cst.RASTER_WEIGHTS_SUM:

            if tag in [cst.RASTER_NB_PTS, cst.RASTER_NB_PTS_IN_CELL]:
                method = "sum"
            elif tag in [
                cst.RASTER_FILLING,
                cst.RASTER_CLASSIF,
                cst.RASTER_SOURCE_PC,
            ]:
                method = "bool"

            else:
                method = "basic"

            old_data, nodata_raster = orchestrator.get_data(tag, future_object)
            current_data = future_object[tag].values
            future_object[tag].values = np.reshape(
                rasterization_step.update_data(
                    old_data,
                    current_data,
                    weights,
                    old_weights,
                    nodata_raster,
                    method=method,
                ),
                current_data.shape,
            )

    return future_object
