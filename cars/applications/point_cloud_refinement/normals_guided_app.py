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
Normals-guided point cloud refinement application.
"""

# pylint: disable=too-many-positional-arguments

import copy
import os

import numpy as np
from json_checker import And, Checker, Or

import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
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

from . import abstract_point_cloud_refinement_app as abstract_refinement
from . import point_cloud_refinement_algo


class NormalsGuidedPointCloudRefinement(
    abstract_refinement.PointCloudRefinement,
    short_name="normals_guided",
):
    """
    Normals guided refinement of triangulated point clouds.
    """

    def __init__(self, conf=None):
        super().__init__(conf=conf)

        self.used_method = self.used_config["method"]
        self.activated = self.used_config["activated"]
        self.save_intermediate_data = self.used_config[
            application_constants.SAVE_INTERMEDIATE_DATA
        ]
        self.w_guidance = self.used_config["w_guidance"]
        self.w_smooth = self.used_config["w_smooth"]
        self.w_data = self.used_config["w_data"]
        self.max_displacement = self.used_config["max_displacement"]
        self.update_xy = self.used_config["update_xy"]
        self.use_metric = self.used_config["use_metric"]
        self.damp = self.used_config["damp"]
        self.max_iter = self.used_config["max_iter"]

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

        overloaded_conf["method"] = conf.get("method", "normals_guided")
        overloaded_conf["activated"] = conf.get("activated", "auto")
        overloaded_conf[application_constants.SAVE_INTERMEDIATE_DATA] = (
            conf.get(application_constants.SAVE_INTERMEDIATE_DATA, False)
        )
        overloaded_conf["w_guidance"] = conf.get("w_guidance", 1.0)
        overloaded_conf["w_smooth"] = conf.get("w_smooth", 0.0)
        overloaded_conf["w_data"] = conf.get("w_data", 0.2)
        overloaded_conf["max_displacement"] = conf.get("max_displacement", 0.5)
        overloaded_conf["update_xy"] = conf.get("update_xy", True)
        overloaded_conf["use_metric"] = conf.get("use_metric", True)
        overloaded_conf["damp"] = conf.get("damp", 1.0e-6)
        overloaded_conf["max_iter"] = conf.get("max_iter", 20)

        schema = {
            "method": str,
            "activated": Or(bool, lambda value: value == "auto"),
            application_constants.SAVE_INTERMEDIATE_DATA: bool,
            "w_guidance": And(Or(int, float), lambda value: value >= 0.0),
            "w_smooth": And(Or(int, float), lambda value: value >= 0.0),
            "w_data": And(Or(int, float), lambda value: value >= 0.0),
            "max_displacement": And(Or(int, float), lambda value: value >= 0.0),
            "update_xy": bool,
            "use_metric": bool,
            "damp": And(Or(int, float), lambda value: value >= 0.0),
            "max_iter": And(int, lambda value: value > 0),
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

        refined = cars_dataset.CarsDataset(
            point_cloud.dataset_type,
            name="point_cloud_refinement_" + pair_key,
        )
        refined.create_empty_copy(point_cloud)
        refined.attributes.update(point_cloud.attributes)

        if isinstance(point_cloud_format, str):
            point_cloud_format = [point_cloud_format]

        if point_cloud_dir is not None and "tif" in point_cloud_format:
            output_dir = os.path.join(point_cloud_dir, "tif")
            point_cloud_export_utils.register_official_point_cloud_tif(
                orchestrator=self.orchestrator,
                output_dir=output_dir,
                point_cloud=refined,
                pair_key=pair_key,
                cars_ds_name_suffix="refined",
                invalidity_dtype="uint8",
            )

        if self.save_intermediate_data:
            output_dir = os.path.join(dump_dir, "tif")
            safe_makedirs(output_dir)

            self.orchestrator.add_to_save_lists(
                os.path.join(output_dir, "X_refined.tif"),
                cst.X,
                refined,
                cars_ds_name="depth_map_x_refined",
                dtype=np.float64,
            )
            self.orchestrator.add_to_save_lists(
                os.path.join(output_dir, "Y_refined.tif"),
                cst.Y,
                refined,
                cars_ds_name="depth_map_y_refined",
                dtype=np.float64,
            )
            self.orchestrator.add_to_save_lists(
                os.path.join(output_dir, "Z_refined.tif"),
                cst.Z,
                refined,
                cars_ds_name="depth_map_z_refined",
                dtype=np.float64,
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(output_dir, "displacement.tif"),
                "displacement",
                refined,
                cars_ds_name="depth_map_displacement",
                dtype=np.float64,
                optional_data=True,
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(output_dir, "invalidity_mask_refined.tif"),
                cst.EPI_INVALIDITY_MASK,
                refined,
                cars_ds_name="depth_map_invalidity_mask_refined",
                dtype="uint8",
                nodata=255,
                optional_data=True,
            )

        return refined

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
            flattened_dataset_name=(
                "point_cloud_refinement_flatten_" + pair_key
            ),
            laz_cars_ds_name="refined_point_cloud_laz_" + pair_key,
            csv_cars_ds_name="refined_point_cloud_csv_" + pair_key,
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
        Run point cloud refinement on epipolar point cloud tiles.
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
                "point_cloud_refinement is deactivated: "
                "skipping all computations"
            )
            return point_cloud

        if point_cloud.dataset_type != "arrays":
            raise RuntimeError(
                "Only arrays dataset type is supported in "
                "point cloud refinement"
            )

        if dump_dir is None:
            dump_dir = os.path.join(
                self.orchestrator.out_dir,
                "dump_dir",
                "point_cloud_refinement",
                pair_key,
            )
        safe_makedirs(dump_dir)

        refined_point_cloud = self._register_output_dataset(
            point_cloud,
            point_cloud_dir,
            dump_dir,
            point_cloud_format,
            pair_key,
        )

        (
            flatten_refined_point_cloud,
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

        [saving_info] = self.orchestrator.get_saving_infos(
            [refined_point_cloud]
        )

        pc_index = None
        if point_cloud_dir:
            pc_index = {}

        for col in range(refined_point_cloud.shape[1]):
            for row in range(refined_point_cloud.shape[0]):
                if point_cloud[row, col] is None:
                    continue

                window = refined_point_cloud.tiling_grid[row, col]
                overlap = refined_point_cloud.overlaps[row, col]
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
                    refined_point_cloud[row, col],
                    flatten_refined_point_cloud[row, col],
                ) = self.orchestrator.cluster.create_task(
                    refine_point_cloud_wrapper,
                    nout=2,
                )(
                    point_cloud[row, col],
                    w_guidance=self.w_guidance,
                    w_smooth=self.w_smooth,
                    w_data=self.w_data,
                    max_displacement=self.max_displacement,
                    update_xy=self.update_xy,
                    use_metric=self.use_metric,
                    damp=self.damp,
                    max_iter=self.max_iter,
                    saving_info=full_saving_info,
                    saving_info_flatten=full_saving_info_flatten,
                    window=window,
                    overlap=overlap,
                    point_cloud_csv_file_name=csv_pc_file_name,
                    point_cloud_laz_file_name=laz_pc_file_name,
                )

        if point_cloud_dir:
            self.orchestrator.update_index(pc_index)

        return refined_point_cloud


# pylint: disable=too-many-positional-arguments
def refine_point_cloud_wrapper(
    point_cloud_tile,
    w_guidance,
    w_smooth,
    w_data,
    max_displacement,
    update_xy,
    use_metric,
    damp,
    max_iter,
    saving_info=None,
    saving_info_flatten=None,
    window=None,
    overlap=None,
    point_cloud_csv_file_name=None,
    point_cloud_laz_file_name=None,
):
    """
    Wrapper used by the orchestrator to refine one tile.
    """

    if not point_cloud_refinement_algo.has_required_bands(point_cloud_tile):
        logger.warning(
            "Point cloud refinement skipped for tile: missing required "
            "edges normals/tile_id bands"
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
        result = point_cloud_refinement_algo.refine_point_cloud_tile(
            point_cloud_tile,
            w_guidance=w_guidance,
            w_smooth=w_smooth,
            w_data=w_data,
            max_displacement=max_displacement,
            update_xy=update_xy,
            use_metric=use_metric,
            damp=damp,
            max_iter=max_iter,
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

    flatten_result = None
    if point_cloud_csv_file_name or point_cloud_laz_file_name:
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
