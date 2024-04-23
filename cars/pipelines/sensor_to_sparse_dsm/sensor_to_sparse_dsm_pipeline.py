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
CARS sensors_to_sparse_dsm pipeline class file
"""

# Standard imports
from __future__ import print_function

import copy
import json
import logging
import os

from pyproj import CRS

# CARS imports
from cars import __version__
from cars.applications.application import Application
from cars.applications.dem_generation import (
    dem_generation_constants as dem_gen_cst,
)
from cars.applications.dem_generation import dem_generation_tools
from cars.applications.grid_generation import grid_correction
from cars.applications.sparse_matching import sparse_matching_tools
from cars.core import roi_tools
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset
from cars.orchestrator import orchestrator
from cars.orchestrator.cluster.log_wrapper import cars_profile
from cars.pipelines.pipeline import Pipeline
from cars.pipelines.pipeline_constants import (
    APPLICATIONS,
    GEOMETRY_PLUGIN,
    INPUTS,
    ORCHESTRATOR,
    OUTPUT,
    PIPELINE,
)
from cars.pipelines.pipeline_template import PipelineTemplate
from cars.pipelines.sensor_to_dense_dsm import dsm_output
from cars.pipelines.sensor_to_dense_dsm import (
    sensor_dense_dsm_constants as sens_cst,
)
from cars.pipelines.sensor_to_dense_dsm import sensors_inputs

# Path in cars package (pkg)
CARS_GEOID_PATH = "geoid/egm96.grd"


@Pipeline.register("sensors_to_sparse_dsm")
class SensorSparseDsmPipeline(PipelineTemplate):
    """
    SensorSparseDsmPipeline
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf, config_json_dir=None):
        """
        Creates pipeline

        :param pipeline_name: name of the pipeline.
        :type pipeline_name: str
        :param cfg: configuration {'matching_cost_method': value}
        :type cfg: dictionary
        :param config_json_dir: path to dir containing json
        :type config_json_dir: str
        """

        # Merge parameters from associated json
        # priority : cars_pipeline.json << user_inputs.json
        # Get root package directory
        package_path = os.path.dirname(__file__)
        json_file = os.path.join(
            package_path,
            "..",
            "conf_pipeline",
            "sensor_to_sparse_dsm.json",
        )
        with open(json_file, "r", encoding="utf8") as fstream:
            pipeline_config = json.load(fstream)

        self.conf = self.merge_pipeline_conf(pipeline_config, conf)

        # check global conf
        self.check_global_schema(self.conf)

        # Used conf
        self.used_conf = {}

        # Prepared config full res
        self.config_full_res = {}

        # Pipeline
        self.used_conf[PIPELINE] = "sensors_to_sparse_dsm"

        # Check conf orchestrator
        self.orchestrator_conf = self.check_orchestrator(
            self.conf.get(ORCHESTRATOR, None)
        )
        self.used_conf[ORCHESTRATOR] = self.orchestrator_conf

        # Check conf inputs
        self.inputs = self.check_inputs(
            self.conf[INPUTS], config_json_dir=config_json_dir
        )

        # Check geometry plugin
        (
            self.inputs,
            self.used_conf[GEOMETRY_PLUGIN],
            self.geom_plugin_without_dem_and_geoid,
            self.geom_plugin_with_dem_and_geoid,
            self.dem_generation_roi,
        ) = sensors_inputs.check_geometry_plugin(
            self.inputs, self.conf.get(GEOMETRY_PLUGIN, None)
        )
        self.used_conf[INPUTS] = self.inputs

        # Get ROI
        (
            self.input_roi_poly,
            self.input_roi_epsg,
        ) = roi_tools.generate_roi_poly_from_inputs(
            self.used_conf[INPUTS][sens_cst.ROI]
        )

        # Check conf output
        self.output = self.check_output(self.conf[OUTPUT])
        self.used_conf[OUTPUT] = self.output

        # Check conf application
        application_conf = self.check_applications(
            self.conf.get(APPLICATIONS, {})
        )
        self.used_conf[APPLICATIONS] = application_conf

        self.config_full_res = copy.deepcopy(self.used_conf)
        self.config_full_res[PIPELINE] = "sensors_to_dense_dsm"
        self.config_full_res.__delitem__("applications")
        self.config_full_res[INPUTS][sens_cst.EPIPOLAR_A_PRIORI] = {}
        self.config_full_res[INPUTS][sens_cst.TERRAIN_A_PRIORI] = {}
        self.config_full_res[INPUTS]["use_epipolar_a_priori"] = True

        # Check conf application vs inputs application
        self.check_inputs_with_applications(self.inputs, application_conf)

    def check_inputs(self, conf, config_json_dir=None):
        """
        Check the inputs given

        :param conf: configuration of inputs
        :type conf: dict
        :param config_json_dir: directory of used json, if
               user filled paths with relative paths
        :type config_json_dir: str

        :return: overloader inputs
        :rtype: dict
        """
        return sensors_inputs.sensors_check_inputs(
            conf, config_json_dir=config_json_dir, check_epipolar_a_priori=False
        )

    def check_output(self, conf):
        """
        Check the output given

        :param conf: configuration of output
        :type conf: dict

        :return: overloader output
        :rtype: dict
        """
        return dsm_output.dense_dsm_check_output(conf)

    @staticmethod
    def check_inputs_with_applications(inputs_conf, application_conf):
        """
        Check for each application the input configuration consistency

        :param inputs_conf: inputs checked configuration
        :type inputs_conf: dict
        :param application_conf: application checked configuration
        :type application_conf: dict
        """

        if "epsg" in inputs_conf and inputs_conf["epsg"]:
            spatial_ref = CRS.from_epsg(inputs_conf["epsg"])
            if spatial_ref.is_geographic:
                if (
                    "point_cloud_rasterization" in application_conf
                    and application_conf["point_cloud_rasterization"][
                        "resolution"
                    ]
                    > 10e-3
                ):
                    logging.warning(
                        "The resolution of the "
                        + "point_cloud_rasterization should be "
                        + "fixed according to the epsg"
                    )

    def check_applications(self, conf):
        """
        Check the given configuration for applications

        :param conf: configuration of applications
        :type conf: dict
        """

        # Check if all specified applications are used
        needed_applications = [
            "grid_generation",
            "sparse_matching",
            "resampling",
            "triangulation",
            "dem_generation",
            "point_cloud_fusion",
            "point_cloud_rasterization",
        ]

        # Initialize used config
        used_conf = {}

        for app_key in conf.keys():
            if app_key not in needed_applications:
                logging.error(
                    "No {} application used in pipeline".format(app_key)
                )
                raise NameError(
                    "No {} application used in pipeline".format(app_key)
                )

        # Epipolar grid generation
        self.epipolar_grid_generation_app = Application(
            "grid_generation", cfg=conf.get("grid_generation", {})
        )
        used_conf["grid_generation"] = (
            self.epipolar_grid_generation_app.get_conf()
        )

        # Sparse Matching
        self.sparse_matching_app = Application(
            "sparse_matching", cfg=conf.get("sparse_matching", {})
        )
        used_conf["sparse_matching"] = self.sparse_matching_app.get_conf()

        # image resampling
        self.resampling_application = Application(
            "resampling", cfg=conf.get("resampling", {})
        )
        used_conf["resampling"] = self.resampling_application.get_conf()

        # Triangulation
        self.triangulation_application = Application(
            "triangulation", cfg=conf.get("triangulation", {})
        )
        used_conf["triangulation"] = self.triangulation_application.get_conf()

        # MNT generation
        self.dem_generation_application = Application(
            "dem_generation", cfg=conf.get("dem_generation", {})
        )
        used_conf["dem_generation"] = self.dem_generation_application.get_conf()

        # Points cloud fusion
        self.pc_fusion_application = Application(
            "point_cloud_fusion", cfg=conf.get("point_cloud_fusion", {})
        )
        used_conf["point_cloud_fusion"] = self.pc_fusion_application.get_conf()

        # Rasterization
        self.rasterization_application = Application(
            "point_cloud_rasterization",
            cfg=conf.get("point_cloud_rasterization", {}),
        )
        used_conf["point_cloud_rasterization"] = (
            self.rasterization_application.get_conf()
        )

        return used_conf

    @cars_profile(name="run_sparse_pipeline", interval=0.5)
    def run(self):
        """
        Run pipeline

        """
        out_dir = self.output["out_dir"]

        # Save used conf
        cars_dataset.save_dict(
            self.used_conf,
            os.path.join(out_dir, "used_conf.json"),
            safe_save=True,
        )

        # start cars orchestrator
        with orchestrator.Orchestrator(
            orchestrator_conf=self.orchestrator_conf,
            out_dir=out_dir,
            out_json_path=os.path.join(
                out_dir, self.output[sens_cst.INFO_BASENAME]
            ),
        ) as cars_orchestrator:
            # initialize out_json
            cars_orchestrator.update_out_info(
                {
                    "version": __version__,
                    "pipeline": "sensors_to_sparse_dsm_pipeline",
                    "inputs": self.inputs,
                }
            )

            # Run applications
            list_sensor_pairs = sensors_inputs.generate_inputs(
                self.inputs, self.geom_plugin_without_dem_and_geoid
            )
            logging.info(
                "Received {} stereo pairs configurations".format(
                    len(list_sensor_pairs)
                )
            )

            pairs = {}
            triangulated_matches_list = []

            for (
                pair_key,
                sensor_image_left,
                sensor_image_right,
            ) in list_sensor_pairs:
                # Create Pair folder
                pair_folder = os.path.join(out_dir, pair_key)
                safe_makedirs(pair_folder)
                safe_makedirs(os.path.join(pair_folder, "tmp"))

                pairs[pair_key] = {}
                pairs[pair_key]["pair_folder"] = pair_folder
                pairs[pair_key]["sensor_image_left"] = sensor_image_left
                pairs[pair_key]["sensor_image_right"] = sensor_image_right

                # Run applications

                # Run grid generation
                if self.inputs[sens_cst.INITIAL_ELEVATION] is None:
                    geom_plugin = self.geom_plugin_without_dem_and_geoid
                else:
                    geom_plugin = self.geom_plugin_with_dem_and_geoid

                (
                    pairs[pair_key]["grid_left"],
                    pairs[pair_key]["grid_right"],
                ) = self.epipolar_grid_generation_app.run(
                    pairs[pair_key]["sensor_image_left"],
                    pairs[pair_key]["sensor_image_right"],
                    geom_plugin,
                    orchestrator=cars_orchestrator,
                    pair_folder=pairs[pair_key]["pair_folder"],
                    pair_key=pair_key,
                )

                # Run epipolar resampling
                (
                    pairs[pair_key]["epipolar_image_left"],
                    pairs[pair_key]["epipolar_image_right"],
                ) = self.resampling_application.run(
                    pairs[pair_key]["sensor_image_left"],
                    pairs[pair_key]["sensor_image_right"],
                    pairs[pair_key]["grid_left"],
                    pairs[pair_key]["grid_right"],
                    orchestrator=cars_orchestrator,
                    pair_folder=pairs[pair_key]["pair_folder"],
                    pair_key=pair_key,
                    margins_fun=self.sparse_matching_app.get_margins_fun(),
                    tile_width=None,
                    tile_height=None,
                    add_color=False,
                )

                # Run epipolar sparse_matching application
                (
                    pairs[pair_key]["epipolar_matches_left"],
                    _,
                ) = self.sparse_matching_app.run(
                    pairs[pair_key]["epipolar_image_left"],
                    pairs[pair_key]["epipolar_image_right"],
                    pairs[pair_key]["grid_left"].attributes[
                        "disp_to_alt_ratio"
                    ],
                    orchestrator=cars_orchestrator,
                    pair_folder=pairs[pair_key]["pair_folder"],
                    pair_key=pair_key,
                )

                # Run cluster breakpoint to compute sifts
                cars_orchestrator.breakpoint()

                # Run grid correction application

                # Filter matches
                matches_array = self.sparse_matching_app.filter_matches(
                    pairs[pair_key]["epipolar_matches_left"],
                    pairs[pair_key]["grid_left"],
                    pairs[pair_key]["grid_right"],
                    orchestrator=cars_orchestrator,
                    pair_key=pair_key,
                    pair_folder=pair_folder,
                    save_matches=self.sparse_matching_app.get_save_matches(),
                )
                # Estimate grid correction
                (
                    pairs[pair_key]["grid_correction_coef"],
                    pairs[pair_key]["corrected_matches_array"],
                    pairs[pair_key]["corrected_matches_cars_ds"],
                    _,
                    _,
                ) = grid_correction.estimate_right_grid_correction(
                    matches_array,
                    pairs[pair_key]["grid_right"],
                    initial_cars_ds=pairs[pair_key]["epipolar_matches_left"],
                    save_matches=self.sparse_matching_app.get_save_matches(),
                    pair_folder=pairs[pair_key]["pair_folder"],
                    pair_key=pair_key,
                    orchestrator=cars_orchestrator,
                )

                # Correct grid right
                pairs[pair_key]["corrected_grid_right"] = (
                    grid_correction.correct_grid(
                        pairs[pair_key]["grid_right"],
                        pairs[pair_key]["grid_correction_coef"],
                        self.epipolar_grid_generation_app.save_grids,
                        pairs[pair_key]["pair_folder"],
                    )
                )

                pairs[pair_key]["corrected_grid_left"] = pairs[pair_key][
                    "grid_left"
                ]

                # Triangulate matches
                pairs[pair_key]["triangulated_matches"] = (
                    dem_generation_tools.triangulate_sparse_matches(
                        pairs[pair_key]["sensor_image_left"],
                        pairs[pair_key]["sensor_image_right"],
                        pairs[pair_key]["grid_left"],
                        pairs[pair_key]["corrected_grid_right"],
                        pairs[pair_key]["corrected_matches_array"],
                        geom_plugin,
                    )
                )
                # filter triangulated_matches
                matches_filter_knn = (
                    self.sparse_matching_app.get_matches_filter_knn()
                )
                matches_filter_dev_factor = (
                    self.sparse_matching_app.get_matches_filter_dev_factor()
                )
                pairs[pair_key]["filtered_triangulated_matches"] = (
                    sparse_matching_tools.filter_point_cloud_matches(
                        pairs[pair_key]["triangulated_matches"],
                        matches_filter_knn=matches_filter_knn,
                        matches_filter_dev_factor=matches_filter_dev_factor,
                    )
                )
                triangulated_matches_list.append(
                    pairs[pair_key]["filtered_triangulated_matches"]
                )

            # Generate MNT from matches
            dem = self.dem_generation_application.run(
                triangulated_matches_list,
                cars_orchestrator.out_dir,
                self.inputs[sens_cst.GEOID],
                dem_roi_to_use=self.dem_generation_roi,
            )
            dem_median = dem.attributes[dem_gen_cst.DEM_MEDIAN_PATH]
            # Generate geometry loader with dem and geoid
            self.geom_plugin_with_dem_and_geoid = (
                sensors_inputs.generate_geometry_plugin_with_dem(
                    self.used_conf[GEOMETRY_PLUGIN],
                    self.inputs,
                    dem=dem_median,
                )
            )
            dem_min = dem.attributes[dem_gen_cst.DEM_MIN_PATH]
            dem_max = dem.attributes[dem_gen_cst.DEM_MAX_PATH]

            # Generate geometry loader with dem and geoid
            self.geom_plugin_with_dem_and_geoid = (
                sensors_inputs.generate_geometry_plugin_with_dem(
                    self.used_conf[GEOMETRY_PLUGIN],
                    self.inputs,
                    dem=dem_median,
                )
            )

            sensors_inputs.update_conf(
                self.config_full_res,
                dem_median=dem_median,
                dem_min=dem_min,
                dem_max=dem_max,
            )

            for pair_key, _, _ in list_sensor_pairs:
                geom_plugin = self.geom_plugin_with_dem_and_geoid
                # Generate grids with new MNT
                (
                    pairs[pair_key]["new_grid_left"],
                    pairs[pair_key]["new_grid_right"],
                ) = self.epipolar_grid_generation_app.run(
                    pairs[pair_key]["sensor_image_left"],
                    pairs[pair_key]["sensor_image_right"],
                    geom_plugin,
                    orchestrator=cars_orchestrator,
                    pair_folder=pairs[pair_key]["pair_folder"],
                    pair_key=pair_key,
                )

                # Correct grids with former matches
                # Transform matches to new grids

                new_grid_matches_array = (
                    AbstractGeometry.transform_matches_from_grids(
                        pairs[pair_key]["corrected_matches_array"],
                        pairs[pair_key]["corrected_grid_left"],
                        pairs[pair_key]["corrected_grid_right"],
                        pairs[pair_key]["new_grid_left"],
                        pairs[pair_key]["new_grid_right"],
                    )
                )

                # Estimate grid_correction
                (
                    pairs[pair_key]["grid_correction_coef"],
                    corrected_matches_array,
                    pairs[pair_key]["corrected_matches_cars_ds"],
                    _,
                    _,
                ) = grid_correction.estimate_right_grid_correction(
                    new_grid_matches_array,
                    pairs[pair_key]["new_grid_right"],
                    initial_cars_ds=pairs[pair_key]["epipolar_matches_left"],
                    save_matches=(self.sparse_matching_app.get_save_matches()),
                    pair_folder=pairs[pair_key]["pair_folder"],
                    pair_key=pair_key,
                    orchestrator=cars_orchestrator,
                )

                # Correct grid right
                pairs[pair_key]["corrected_grid_right"] = (
                    grid_correction.correct_grid(
                        pairs[pair_key]["new_grid_right"],
                        pairs[pair_key]["grid_correction_coef"],
                        self.epipolar_grid_generation_app.save_grids,
                        pairs[pair_key]["pair_folder"],
                    )
                )
                pairs[pair_key]["corrected_grid_left"] = pairs[pair_key][
                    "new_grid_left"
                ]

                # Triangulate new matches
                pairs[pair_key]["triangulated_matches"] = (
                    dem_generation_tools.triangulate_sparse_matches(
                        pairs[pair_key]["sensor_image_left"],
                        pairs[pair_key]["sensor_image_right"],
                        pairs[pair_key]["corrected_grid_left"],
                        pairs[pair_key]["corrected_grid_right"],
                        corrected_matches_array,
                        geometry_plugin=geom_plugin,
                    )
                )

                # filter triangulated_matches
                matches_filter_knn = (
                    self.sparse_matching_app.get_matches_filter_knn()
                )
                matches_filter_dev_factor = (
                    self.sparse_matching_app.get_matches_filter_dev_factor()
                )
                pairs[pair_key]["filtered_triangulated_matches"] = (
                    sparse_matching_tools.filter_point_cloud_matches(
                        pairs[pair_key]["triangulated_matches"],
                        matches_filter_knn=matches_filter_knn,
                        matches_filter_dev_factor=matches_filter_dev_factor,
                    )
                )

                # Compute disp_min and disp_max
                (dmin, dmax) = sparse_matching_tools.compute_disp_min_disp_max(
                    pairs[pair_key]["filtered_triangulated_matches"],
                    cars_orchestrator,
                    disp_margin=(
                        self.sparse_matching_app.get_disparity_margin()
                    ),
                    pair_key=pair_key,
                    disp_to_alt_ratio=pairs[pair_key][
                        "corrected_grid_left"
                    ].attributes["disp_to_alt_ratio"],
                )

                # Update full res pipeline configuration
                # with grid correction and disparity range
                sensors_inputs.update_conf(
                    self.config_full_res,
                    grid_correction_coef=pairs[pair_key][
                        "grid_correction_coef"
                    ],
                    dmin=dmin,
                    dmax=dmax,
                    pair_key=pair_key,
                )

            # Save the refined full res pipeline configuration
            cars_dataset.save_dict(
                self.config_full_res,
                os.path.join(out_dir, "refined_config_dense_dsm.json"),
                safe_save=True,
            )
