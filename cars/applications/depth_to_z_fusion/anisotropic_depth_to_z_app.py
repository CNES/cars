#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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
Anisotropic depth to Z fusion application.
"""

# pylint: disable=too-many-positional-arguments

import copy
import os

import numpy as np
from json_checker import And, Checker, Or

import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.depth_to_z_fusion import (
    abstract_depth_to_z_fusion_app as abstract_fusion,
)
from cars.applications.depth_to_z_fusion import (
    depth_to_z_fusion_algo,
)
from cars.applications.triangulation import (
    point_cloud_export_utils,
)
from cars.applications.triangulation import (
    triangulation_wrappers as triangulation_wrap,
)
from cars.core import constants as cst
from cars.core.cars_logging import logger
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset


class AnisotropicDepthToZFusion(
    abstract_fusion.DepthToZFusion,
    short_name="anisotropic",
):
    """
    Fit monocular depth structure onto stereo Z through anisotropic fusion.
    """

    def __init__(self, conf=None):
        super().__init__(conf=conf)

        self.used_method = self.used_config["method"]
        self.activated = self.used_config["activated"]
        self.save_intermediate_data = self.used_config[
            application_constants.SAVE_INTERMEDIATE_DATA
        ]
        self.lambda_data = self.used_config["lambda_data"]
        self.depth_sigma = self.used_config["depth_sigma"]
        self.depth_sigma_auto = self.used_config["depth_sigma_auto"]
        self.detail_scale = self.used_config["detail_scale"]
        self.cross_tile_weight = self.used_config["cross_tile_weight"]
        self.fill_values = self.used_config["fill_values"]
        self.iterations = self.used_config["iterations"]
        self.tol = self.used_config["tol"]

        self.orchestrator = None

    def check_conf(self, conf):
        """
        Check configuration.
        """

        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        overloaded_conf["method"] = conf.get("method", "anisotropic")
        overloaded_conf["activated"] = conf.get("activated", "auto")
        overloaded_conf[application_constants.SAVE_INTERMEDIATE_DATA] = (
            conf.get(application_constants.SAVE_INTERMEDIATE_DATA, False)
        )
        overloaded_conf["lambda_data"] = float(conf.get("lambda_data", 0.02))
        overloaded_conf["depth_sigma"] = conf.get("depth_sigma", None)
        overloaded_conf["depth_sigma_auto"] = conf.get("depth_sigma_auto", True)
        overloaded_conf["detail_scale"] = float(conf.get("detail_scale", 1.0))
        overloaded_conf["cross_tile_weight"] = float(
            conf.get("cross_tile_weight", 1.0)
        )
        overloaded_conf["fill_values"] = conf.get("fill_values", "invalid")
        overloaded_conf["iterations"] = int(conf.get("iterations", 50))
        overloaded_conf["tol"] = float(conf.get("tol", 1.0e-4))

        if overloaded_conf["fill_values"] not in (None, "invalid", "all"):
            raise ValueError(
                "fill_values must be one of None, 'invalid', or 'all'"
            )

        schema = {
            "method": str,
            "activated": Or(bool, lambda value: value == "auto"),
            application_constants.SAVE_INTERMEDIATE_DATA: bool,
            "lambda_data": And(float, lambda value: value >= 0.0),
            "depth_sigma": And(
                Or(None, int, float),
                lambda value: value is None or value > 0.0,
            ),
            "depth_sigma_auto": bool,
            "detail_scale": float,
            "cross_tile_weight": And(float, lambda value: value >= 0.0),
            "fill_values": Or(None, str),
            "iterations": And(int, lambda value: value > 0),
            "tol": And(float, lambda value: value > 0.0),
        }

        Checker(schema).validate(overloaded_conf)
        return overloaded_conf

    def default_value_for_auto_configuration(self):
        """
        Update auto configuration values with their defaults.
        """

        if self.activated == "auto":
            self.activated = True

    def _register_output_dataset(
        self,
        point_cloud,
        point_cloud_dir,
        dump_dir,
        point_cloud_format,
        pair_key,
    ):
        """
        Create and register output CarsDataset.
        """

        fused = cars_dataset.CarsDataset(
            point_cloud.dataset_type,
            name="depth_to_z_fusion_" + pair_key,
        )
        fused.create_empty_copy(point_cloud)
        fused.attributes.update(point_cloud.attributes)

        if isinstance(point_cloud_format, str):
            point_cloud_format = [point_cloud_format]

        if point_cloud_dir is not None and "tif" in point_cloud_format:
            output_dir = os.path.join(point_cloud_dir, "tif")
            point_cloud_export_utils.register_official_point_cloud_tif(
                orchestrator=self.orchestrator,
                output_dir=output_dir,
                point_cloud=fused,
                pair_key=pair_key,
                cars_ds_name_suffix="fused",
                invalidity_dtype=np.uint8,
            )

        if self.save_intermediate_data:
            output_dir = os.path.join(dump_dir, "tif")
            safe_makedirs(output_dir)

            self.orchestrator.add_to_save_lists(
                os.path.join(output_dir, "fit_depth_map.tif"),
                "fit_depth_map",
                fused,
                cars_ds_name="depth_map_fit_depth_map",
                dtype=np.float32,
            )
            self.orchestrator.add_to_save_lists(
                os.path.join(output_dir, "residual.tif"),
                "depth_to_z_residual",
                fused,
                cars_ds_name="depth_map_depth_to_z_residual",
                dtype=np.float32,
                optional_data=True,
            )
            self.orchestrator.add_to_save_lists(
                os.path.join(output_dir, "weight_map.tif"),
                "depth_to_z_weight_map",
                fused,
                cars_ds_name="depth_map_depth_to_z_weight_map",
                dtype=np.float32,
                optional_data=True,
            )
            self.orchestrator.add_to_save_lists(
                os.path.join(output_dir, "filling.tif"),
                cst.EPI_FILLING,
                fused,
                cars_ds_name="depth_map_filling",
                dtype=np.uint8,
                optional_data=True,
                nodata=255,
            )

        return fused

    def _register_output_point_cloud_dataset(
        self,
        point_cloud,
        point_cloud_dir,
        dump_dir,
        point_cloud_format,
        pair_key,
    ):
        """
        Create and register flattened point cloud output dataset.
        """
        return point_cloud_export_utils.register_flatten_point_cloud_dataset(
            orchestrator=self.orchestrator,
            point_cloud=point_cloud,
            point_cloud_dir=point_cloud_dir,
            dump_dir=dump_dir,
            point_cloud_format=point_cloud_format,
            save_intermediate_data=self.save_intermediate_data,
            flattened_dataset_name=("depth_to_z_fusion_flatten_" + pair_key),
            laz_cars_ds_name="fused_point_cloud_laz_" + pair_key,
            csv_cars_ds_name="fused_point_cloud_csv_" + pair_key,
        )

    def run(
        self,
        point_cloud,
        orchestrator=None,
        point_cloud_dir=None,
        point_cloud_format="laz",
        dump_dir=None,
        pair_key="PAIR_0",
    ):
        """
        Run depth to Z fusion on epipolar point cloud tiles.
        """

        self.default_value_for_auto_configuration()

        if orchestrator is None:
            self.orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )
        else:
            self.orchestrator = orchestrator

        if not self.activated:
            logger.debug(
                "depth_to_z_fusion is deactivated: skipping all computations"
            )
            return point_cloud

        if point_cloud.dataset_type != "arrays":
            raise RuntimeError(
                "Only arrays dataset type is supported in depth to Z fusion"
            )

        if dump_dir is None:
            dump_dir = os.path.join(
                self.orchestrator.out_dir,
                "dump_dir",
                "depth_to_z_fusion",
                pair_key,
            )
        safe_makedirs(dump_dir)

        fused_point_cloud = self._register_output_dataset(
            point_cloud,
            point_cloud_dir,
            dump_dir,
            point_cloud_format,
            pair_key,
        )

        (
            flatten_fused_point_cloud,
            laz_pc_dir_name,
            csv_pc_dir_name,
            saving_info_flatten,
        ) = self._register_output_point_cloud_dataset(
            point_cloud,
            point_cloud_dir,
            dump_dir,
            point_cloud_format,
            pair_key,
        )

        [saving_info] = self.orchestrator.get_saving_infos([fused_point_cloud])

        pc_index = None
        if point_cloud_dir:
            pc_index = {}

        for col in range(fused_point_cloud.shape[1]):
            for row in range(fused_point_cloud.shape[0]):
                if point_cloud[row, col] is None:
                    continue

                window = fused_point_cloud.tiling_grid[row, col]
                overlap = fused_point_cloud.overlaps[row, col]
                full_saving_info = ocht.update_saving_infos(
                    saving_info,
                    row=row,
                    col=col,
                )
                full_saving_info_flatten = ocht.update_saving_infos(
                    saving_info_flatten,
                    row=row,
                    col=col,
                )
                csv_pc_file_name, laz_pc_file_name = (
                    triangulation_wrap.generate_point_cloud_file_names(
                        csv_pc_dir_name,
                        laz_pc_dir_name,
                        row,
                        col,
                        pc_index,
                        pair_key,
                    )
                )

                (
                    fused_point_cloud[row, col],
                    flatten_fused_point_cloud[row, col],
                ) = self.orchestrator.cluster.create_task(
                    fuse_depth_to_z_wrapper,
                    nout=2,
                )(
                    point_cloud[row, col],
                    lambda_data=self.lambda_data,
                    depth_sigma=self.depth_sigma,
                    depth_sigma_auto=self.depth_sigma_auto,
                    detail_scale=self.detail_scale,
                    cross_tile_weight=self.cross_tile_weight,
                    fill_values=self.fill_values,
                    iterations=self.iterations,
                    tol=self.tol,
                    saving_info=full_saving_info,
                    saving_info_flatten=full_saving_info_flatten,
                    window=window,
                    overlap=overlap,
                    point_cloud_csv_file_name=csv_pc_file_name,
                    point_cloud_laz_file_name=laz_pc_file_name,
                )

        if point_cloud_dir:
            self.orchestrator.update_index(pc_index)

        return fused_point_cloud


def fuse_depth_to_z_wrapper(
    point_cloud_tile,
    lambda_data,
    depth_sigma,
    depth_sigma_auto,
    detail_scale,
    cross_tile_weight,
    fill_values,
    iterations,
    tol,
    saving_info=None,
    saving_info_flatten=None,
    window=None,
    overlap=None,
    point_cloud_csv_file_name=None,
    point_cloud_laz_file_name=None,
):
    """
    Wrapper used by the orchestrator to fuse one tile.
    """

    if not depth_to_z_fusion_algo.has_required_bands(point_cloud_tile):
        logger.warning(
            "Depth to Z fusion skipped for tile: missing required "
            "edges depth map/tile_id bands"
        )
        result = point_cloud_tile.copy(deep=False)
        cars_dataset.fill_dataset(
            result,
            saving_info=saving_info,
            window=cars_dataset.window_array_to_dict(window),
            profile=cars_dataset.get_profile_rasterio(point_cloud_tile),
            attributes=copy.deepcopy(
                cars_dataset.get_attributes(point_cloud_tile)
            ),
            overlaps=cars_dataset.overlap_array_to_dict(overlap),
        )
    else:
        result = depth_to_z_fusion_algo.fit_depth_to_z_tile(
            point_cloud_tile,
            lambda_data=lambda_data,
            depth_sigma=depth_sigma,
            depth_sigma_auto=depth_sigma_auto,
            detail_scale=detail_scale,
            cross_tile_weight=cross_tile_weight,
            fill_values=fill_values,
            iterations=iterations,
            tol=tol,
        )

        attributes = cars_dataset.get_attributes(point_cloud_tile)

        cars_dataset.fill_dataset(
            result,
            saving_info=saving_info,
            window=cars_dataset.window_array_to_dict(window),
            profile=cars_dataset.get_profile_rasterio(point_cloud_tile),
            attributes=copy.deepcopy(attributes),
            overlaps=cars_dataset.overlap_array_to_dict(overlap),
        )

    save_point_cloud = any(
        path is not None
        for path in (
            point_cloud_csv_file_name,
            point_cloud_laz_file_name,
        )
    )
    if not save_point_cloud:
        return result, None

    flatten_result = point_cloud_export_utils.build_flattened_dataframe(
        result=result,
        point_cloud_tile=point_cloud_tile,
        saving_info_flatten=saving_info_flatten,
    )

    point_cloud_export_utils.save_flattened_dataframe(
        flatten_result=flatten_result,
        point_cloud_csv_file_name=point_cloud_csv_file_name,
        point_cloud_laz_file_name=point_cloud_laz_file_name,
    )

    return result, flatten_result
