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
CARS sensors_to_dense_dsm pipeline class file
"""
# Standard imports
from __future__ import print_function

import json
import logging
import os

import numpy as np

# CARS imports
from cars import __version__
from cars.applications.application import Application
from cars.applications.dem_generation import (
    dem_generation_constants as dem_gen_cst,
)
from cars.applications.dem_generation import dem_generation_tools
from cars.applications.grid_generation import grid_correction
from cars.applications.sparse_matching import (
    sparse_matching_tools as sparse_mtch_tools,
)
from cars.core import constants_disparity as cst_disp
from cars.core import preprocessing, roi_tools
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.core.inputs import get_descriptions_bands
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset
from cars.orchestrator import orchestrator
from cars.orchestrator.cluster.log_wrapper import cars_profile
from cars.pipelines.parameters import advanced_parameters
from cars.pipelines.parameters import advanced_parameters_constants as adv_cst
from cars.pipelines.parameters import (
    output_constants,
    output_parameters,
    sensor_inputs,
)
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from cars.pipelines.pipeline import Pipeline
from cars.pipelines.pipeline_constants import (
    ADVANCED,
    APPLICATIONS,
    GEOMETRY_PLUGIN,
    INPUTS,
    ORCHESTRATOR,
    OUTPUT,
    PIPELINE,
)
from cars.pipelines.pipeline_template import PipelineTemplate


@Pipeline.register(
    "sensors_to_dense_dsm",
    "sensors_to_dense_dsm_no_merging",
    "sensors_to_dense_depth_maps",
)
class SensorToDenseDsmPipeline(PipelineTemplate):
    """
    SensorToDenseDsmPipeline
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

        # Used conf
        self.used_conf = {}

        # Pipeline
        self.used_conf[PIPELINE] = conf.get(
            PIPELINE, "sensors_to_dense_dsm_no_merging"
        )

        self.generate_terrain_products = True
        # set json pipeline file
        if "sensors_to_dense_dsm" in self.used_conf[PIPELINE]:
            json_conf_file_name = "sensor_to_dense_dsm.json"
        else:
            json_conf_file_name = "sensor_to_pc.json"
            self.generate_terrain_products = False

        # Merge parameters from associated json
        # priority : cars_pipeline.json << user_inputs.json
        # Get root package directory
        package_path = os.path.dirname(__file__)
        json_file = os.path.join(
            package_path,
            "..",
            "conf_pipeline",
            json_conf_file_name,
        )
        with open(json_file, "r", encoding="utf8") as fstream:
            pipeline_config = json.load(fstream)

        self.conf = self.merge_pipeline_conf(pipeline_config, conf)

        # check global conf
        self.check_global_schema(self.conf)

        # Check conf orchestrator
        self.orchestrator_conf = self.check_orchestrator(
            self.conf.get(ORCHESTRATOR, None)
        )
        self.used_conf[ORCHESTRATOR] = self.orchestrator_conf

        # Check conf inputs
        self.inputs = self.check_inputs(
            self.conf[INPUTS], config_json_dir=config_json_dir
        )

        # Check advanced parameters
        # TODO static method in the base class
        self.advanced = advanced_parameters.check_advanced_parameters(
            self.conf.get(ADVANCED, {}), check_epipolar_a_priori=True
        )
        self.used_conf[ADVANCED] = self.advanced

        # Check geometry plugin and overwrite geomodel in conf inputs
        (
            self.inputs,
            self.used_conf[GEOMETRY_PLUGIN],
            self.geom_plugin_without_dem_and_geoid,
            self.geom_plugin_with_dem_and_geoid,
            self.dem_generation_roi,
        ) = sensor_inputs.check_geometry_plugin(
            self.inputs, self.advanced, self.conf.get(GEOMETRY_PLUGIN, None)
        )
        self.used_conf[INPUTS] = self.inputs

        # Get ROI
        (
            self.input_roi_poly,
            self.input_roi_epsg,
        ) = roi_tools.generate_roi_poly_from_inputs(
            self.used_conf[INPUTS][sens_cst.ROI]
        )

        self.debug_with_roi = self.used_conf[ADVANCED][adv_cst.DEBUG_WITH_ROI]
        # Check conf output
        self.output = self.check_output(
            self.conf[OUTPUT], self.used_conf[PIPELINE]
        )
        self.used_conf[OUTPUT] = self.output

        self.save_output_dsm = (
            "dsm" in self.output[output_constants.PRODUCT_LEVEL]
        )
        self.save_output_depth_map = (
            "depth_map" in self.output[output_constants.PRODUCT_LEVEL]
        )
        self.save_output_point_cloud = (
            "point_cloud" in self.output[output_constants.PRODUCT_LEVEL]
        )

        # Check conf application
        self.application_conf = self.check_applications(
            self.conf.get(APPLICATIONS, {}),
            self.generate_terrain_products,
            no_merging="no_merging" in self.used_conf[PIPELINE],
            save_all_intermediate_data=self.used_conf[ADVANCED][
                adv_cst.SAVE_INTERMEDIATE_DATA
            ],
            save_all_point_clouds_by_pair=self.used_conf[OUTPUT].get(
                output_constants.SAVE_BY_PAIR, False
            ),
        )

        # Check conf application vs inputs application
        self.application_conf = self.check_applications_with_inputs(
            self.inputs, self.application_conf
        )

        self.used_conf[APPLICATIONS] = self.application_conf

    @staticmethod
    def check_inputs(conf, config_json_dir=None):
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
        return sensor_inputs.sensors_check_inputs(
            conf, config_json_dir=config_json_dir
        )

    @staticmethod
    def check_output(conf, pipeline):
        """
        Check the output given

        :param conf: configuration of output
        :type conf: dict
        :param pipeline: name of corresponding pipeline
        :type pipeline_name: str

        :return overloader output
        :rtype : dict
        """
        return output_parameters.check_output_parameters(conf, pipeline)

    def check_applications(
        self,
        conf,
        generate_terrain_products,
        no_merging=False,
        save_all_intermediate_data=False,
        save_all_point_clouds_by_pair=False,
    ):
        """
        Check the given configuration for applications,
        and generates needed applications for pipeline.

        :param conf: configuration of applications
        :type conf: dict
        :param generate_terrain_products: true if uses point cloud
            fusion, pc removing, rasterization
        :type generate_terrain_products: bool
        :param no_merging: True if skip PC fusion and PC removing
        :type no_merging: bool
        :param save_all_intermediate_data: True to save intermediate data in all
            applications
        :type save_all_intermediate_data: bool
        :param save_all_point_clouds_by_pair: save point clouds by pair in all
            relevant applications
        :type save_all_point_clouds_by_pair: bool
        """

        # Check if all specified applications are used
        # Application in terrain_application are note used in
        # the sensors_to_dense_depth_maps pipeline
        needed_applications = [
            "grid_generation",
            "resampling",
            "holes_detection",
            "dense_matches_filling.1",
            "dense_matches_filling.2",
            "sparse_matching",
            "dense_matching",
            "triangulation",
            "pc_denoising",
            "dem_generation",
        ]

        terrain_applications = [
            "point_cloud_rasterization",
        ]

        if not no_merging:
            terrain_applications.append("point_cloud_fusion")
            terrain_applications.append("point_cloud_outliers_removing.1")
            terrain_applications.append("point_cloud_outliers_removing.2")

        pipeline_name = "sensors_to_dense_depth_maps"
        if generate_terrain_products:
            needed_applications += terrain_applications
            pipeline_name = "sensors_to_dense_dsm"
            if no_merging:
                pipeline_name += "_no_merging"

        for app_key in conf.keys():
            if app_key not in needed_applications:
                logging.error(
                    "No {} application used in pipeline {}".format(
                        app_key, pipeline_name
                    )
                )
                raise NameError(
                    "No {} application used in pipeline {}".format(
                        app_key, pipeline_name
                    )
                )

        # Initialize used config
        used_conf = {}
        for app_key in needed_applications:
            used_conf[app_key] = conf.get(app_key, {})
            used_conf[app_key]["save_intermediate_data"] = used_conf[
                app_key
            ].get("save_intermediate_data", save_all_intermediate_data)

        for app_key in [
            "point_cloud_fusion",
            "point_cloud_outliers_removing.1",
            "point_cloud_outliers_removing.2",
            "pc_denoising",
        ]:
            if app_key in needed_applications:
                used_conf[app_key]["save_by_pair"] = used_conf[app_key].get(
                    "save_by_pair", save_all_point_clouds_by_pair
                )

        # Epipolar grid generation
        self.epipolar_grid_generation_application = Application(
            "grid_generation", cfg=used_conf.get("grid_generation", {})
        )
        used_conf["grid_generation"] = (
            self.epipolar_grid_generation_application.get_conf()
        )

        # image resampling
        self.resampling_application = Application(
            "resampling", cfg=used_conf.get("resampling", {})
        )
        used_conf["resampling"] = self.resampling_application.get_conf()

        # holes detection
        self.holes_detection_app = Application(
            "holes_detection", cfg=used_conf.get("holes_detection", {})
        )
        used_conf["holes_detection"] = self.holes_detection_app.get_conf()

        # disparity filling 1 plane
        self.dense_matches_filling_1 = Application(
            "dense_matches_filling",
            cfg=used_conf.get(
                "dense_matches_filling.1",
                {"method": "plane"},
            ),
        )
        used_conf["dense_matches_filling.1"] = (
            self.dense_matches_filling_1.get_conf()
        )

        # disparity filling 2
        self.dense_matches_filling_2 = Application(
            "dense_matches_filling",
            cfg=used_conf.get(
                "dense_matches_filling.2",
                {"method": "zero_padding"},
            ),
        )
        used_conf["dense_matches_filling.2"] = (
            self.dense_matches_filling_2.get_conf()
        )

        # Sparse Matching
        self.sparse_mtch_app = Application(
            "sparse_matching", cfg=used_conf.get("sparse_matching", {})
        )
        used_conf["sparse_matching"] = self.sparse_mtch_app.get_conf()

        # Matching
        generate_performance_map = (
            self.used_conf[OUTPUT]
            .get(output_constants.AUXILIARY, {})
            .get(output_constants.AUX_PERFORMANCE_MAP, False)
        )
        dense_matching_config = used_conf.get("dense_matching", {})
        if generate_performance_map is True:
            dense_matching_config["generate_performance_map"] = True
        self.dense_matching_app = Application(
            "dense_matching", cfg=dense_matching_config
        )
        used_conf["dense_matching"] = self.dense_matching_app.get_conf()

        # Triangulation
        self.triangulation_application = Application(
            "triangulation", cfg=used_conf.get("triangulation", {})
        )
        used_conf["triangulation"] = self.triangulation_application.get_conf()

        self.pc_denoising_application = Application(
            "pc_denoising",
            cfg=used_conf.get("pc_denoising", {"method": "none"}),
        )

        # MNT generation
        self.dem_generation_application = Application(
            "dem_generation", cfg=used_conf.get("dem_generation", {})
        )
        used_conf["dem_generation"] = self.dem_generation_application.get_conf()

        if generate_terrain_products:
            # Points cloud fusion
            self.pc_fusion_application = Application(
                "point_cloud_fusion",
                cfg=used_conf.get("point_cloud_fusion", {}),
            )
            if not no_merging:
                used_conf["point_cloud_fusion"] = (
                    self.pc_fusion_application.get_conf()
                )

            # Points cloud outlier removing small components
            self.pc_outliers_removing_1_app = Application(
                "point_cloud_outliers_removing",
                cfg=used_conf.get(
                    "point_cloud_outliers_removing.1",
                    {"method": "small_components"},
                ),
            )
            if not no_merging:
                used_conf["point_cloud_outliers_removing.1"] = (
                    self.pc_outliers_removing_1_app.get_conf()
                )

            # Points cloud outlier removing statistical
            self.pc_outliers_removing_2_app = Application(
                "point_cloud_outliers_removing",
                cfg=used_conf.get(
                    "point_cloud_outliers_removing.2",
                    {"method": "statistical"},
                ),
            )
            if not no_merging:
                used_conf["point_cloud_outliers_removing.2"] = (
                    self.pc_outliers_removing_2_app.get_conf()
                )

            # Rasterization
            self.rasterization_application = Application(
                "point_cloud_rasterization",
                cfg=used_conf.get("point_cloud_rasterization", {}),
            )
            used_conf["point_cloud_rasterization"] = (
                self.rasterization_application.get_conf()
            )
        else:
            # Points cloud fusion
            self.pc_fusion_application = None
            # Points cloud outlier removing small components
            self.pc_outliers_removing_1_app = None
            # Points cloud outlier removing statistical
            self.pc_outliers_removing_2_app = None
            # Rasterization
            self.rasterization_application = None

        return used_conf

    def check_applications_with_inputs(self, inputs_conf, application_conf):
        """
        Check for each application the input and output configuration
        consistency

        :param inputs_conf: inputs checked configuration
        :type inputs_conf: dict
        :param application_conf: application checked configuration
        :type application_conf: dict
        """

        initial_elevation = self.inputs["initial_elevation"] is not None
        if self.sparse_mtch_app.elevation_delta_lower_bound is None:
            self.sparse_mtch_app.used_config["elevation_delta_lower_bound"] = (
                -100 if initial_elevation else -1000
            )
            self.sparse_mtch_app.elevation_delta_lower_bound = (
                self.sparse_mtch_app.used_config["elevation_delta_lower_bound"]
            )
        if self.sparse_mtch_app.elevation_delta_upper_bound is None:
            self.sparse_mtch_app.used_config["elevation_delta_upper_bound"] = (
                1000 if initial_elevation else 9000
            )
            self.sparse_mtch_app.elevation_delta_upper_bound = (
                self.sparse_mtch_app.used_config["elevation_delta_upper_bound"]
            )
        application_conf["sparse_matching"] = self.sparse_mtch_app.get_conf()

        # check classification application parameter compare
        # to each sensors inputs classification list
        for application_key in application_conf:
            if "classification" in application_conf[application_key]:
                for item in inputs_conf["sensors"]:
                    if "classification" in inputs_conf["sensors"][item].keys():
                        if inputs_conf["sensors"][item]["classification"]:
                            descriptions = get_descriptions_bands(
                                inputs_conf["sensors"][item]["classification"]
                            )
                            if application_conf[application_key][
                                "classification"
                            ] and not set(
                                application_conf[application_key][
                                    "classification"
                                ]
                            ).issubset(
                                set(descriptions)
                            ):
                                raise RuntimeError(
                                    "The {} bands description {} ".format(
                                        inputs_conf["sensors"][item][
                                            "classification"
                                        ],
                                        list(descriptions),
                                    )
                                    + "and the {} config are not ".format(
                                        application_key
                                    )
                                    + "consistent: {}".format(
                                        application_conf[application_key][
                                            "classification"
                                        ]
                                    )
                                )
        for key1, key2 in inputs_conf["pairing"]:
            corr_cfg = self.dense_matching_app.loader.get_conf()
            img_left = inputs_conf["sensors"][key1]["image"]
            img_right = inputs_conf["sensors"][key2]["image"]
            classif_left = None
            classif_right = None
            if "classification" in inputs_conf["sensors"][key1]:
                classif_left = inputs_conf["sensors"][key1]["classification"]
            if "classification" in inputs_conf["sensors"][key2]:
                classif_right = inputs_conf["sensors"][key2]["classification"]
            self.dense_matching_app.corr_config = (
                self.dense_matching_app.loader.check_conf(
                    corr_cfg,
                    img_left,
                    img_right,
                    classif_left,
                    classif_right,
                )
            )

        return application_conf

    @cars_profile(name="run_dense_pipeline", interval=0.5)
    def run(self):  # noqa C901
        """
        Run pipeline

        """

        out_dir = self.output[output_constants.OUT_DIRECTORY]
        auxiliary = self.output[output_constants.AUXILIARY]

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
                out_dir,
                output_constants.INFO_FILENAME,
            ),
        ) as cars_orchestrator:
            # initialize out_json
            cars_orchestrator.update_out_info(
                {
                    "version": __version__,
                    "pipeline": "sensor_to_dense_dsm_pipeline",
                    "inputs": self.inputs,
                }
            )

            # Application dump directory
            dump_dir = os.path.join(cars_orchestrator.out_dir, "dump_dir")

            # Run applications

            # Initialize epsg for terrain tiles
            epsg = self.output[output_constants.EPSG]
            if epsg is not None:
                # Compute roi polygon, in output EPSG
                roi_poly = preprocessing.compute_roi_poly(
                    self.input_roi_poly, self.input_roi_epsg, epsg
                )

            resolution = self.output[output_constants.RESOLUTION]

            # List of terrain roi corresponding to each epipolar pair
            # Used to generate final terrain roi
            list_terrain_roi = []

            # initialise lists of points
            list_epipolar_points_cloud = []
            list_sensor_pairs = sensor_inputs.generate_inputs(
                self.inputs, self.geom_plugin_without_dem_and_geoid
            )
            logging.info(
                "Received {} stereo pairs configurations".format(
                    len(list_sensor_pairs)
                )
            )

            # pairs is a dict used to store the CarsDataset of
            # all pairs, easily retrievable with pair keys
            pairs = {}

            # triangulated_matches_list is used to store triangulated matche
            # used in dem generation
            triangulated_matches_list = []

            for (
                pair_key,
                sensor_image_left,
                sensor_image_right,
            ) in list_sensor_pairs:
                # initialize pairs for current pair
                pairs[pair_key] = {}
                pairs[pair_key]["sensor_image_left"] = sensor_image_left
                pairs[pair_key]["sensor_image_right"] = sensor_image_right

                # Run applications

                # Run grid generation
                # We generate grids with dem if it is provided.
                # If not provided, grid are generated without dem and a dem
                # will be generated, to use later for a new grid generation**

                if (
                    self.inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]
                    is None
                ):
                    geom_plugin = self.geom_plugin_without_dem_and_geoid
                else:
                    geom_plugin = self.geom_plugin_with_dem_and_geoid

                # Generate rectification grids
                (
                    pairs[pair_key]["grid_left"],
                    pairs[pair_key]["grid_right"],
                ) = self.epipolar_grid_generation_application.run(
                    pairs[pair_key]["sensor_image_left"],
                    pairs[pair_key]["sensor_image_right"],
                    geom_plugin,
                    orchestrator=cars_orchestrator,
                    pair_folder=os.path.join(
                        dump_dir,
                        "epipolar_grid_generation",
                        "initial",
                        pair_key,
                    ),
                    pair_key=pair_key,
                )

                # Run holes detection
                # Get classif depending on which filling is used
                # For now, 2 filling application can be used, and be configured
                # with any order. the .1 will be performed before the .2
                pairs[pair_key]["holes_classif"] = []
                pairs[pair_key]["holes_poly_margin"] = 0
                if self.dense_matches_filling_1.used_method == "plane":
                    pairs[pair_key][
                        "holes_classif"
                    ] += self.dense_matches_filling_1.get_classif()
                    pairs[pair_key]["holes_poly_margin"] = max(
                        pairs[pair_key]["holes_poly_margin"],
                        self.dense_matches_filling_1.get_poly_margin(),
                    )
                if self.dense_matches_filling_2.used_method == "plane":
                    pairs[pair_key][
                        "holes_classif"
                    ] += self.dense_matches_filling_2.get_classif()
                    pairs[pair_key]["holes_poly_margin"] = max(
                        pairs[pair_key]["holes_poly_margin"],
                        self.dense_matches_filling_2.get_poly_margin(),
                    )

                pairs[pair_key]["holes_bbox_left"] = []
                pairs[pair_key]["holes_bbox_right"] = []

                if self.used_conf[ADVANCED][
                    adv_cst.USE_EPIPOLAR_A_PRIORI
                ] is False or (len(pairs[pair_key]["holes_classif"]) > 0):
                    # Run resampling only if needed:
                    # no a priori or needs to detect holes

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
                        pair_folder=os.path.join(
                            dump_dir, "resampling", "initial", pair_key
                        ),
                        pair_key=pair_key,
                        margins_fun=self.sparse_mtch_app.get_margins_fun(),
                        tile_width=None,
                        tile_height=None,
                        add_color=False,
                    )

                    # Generate the holes polygons in epipolar images
                    # They are only generated if dense_matches_filling
                    # applications are used later
                    (
                        pairs[pair_key]["holes_bbox_left"],
                        pairs[pair_key]["holes_bbox_right"],
                    ) = self.holes_detection_app.run(
                        pairs[pair_key]["epipolar_image_left"],
                        pairs[pair_key]["epipolar_image_right"],
                        classification=pairs[pair_key]["holes_classif"],
                        margin=pairs[pair_key]["holes_poly_margin"],
                        orchestrator=cars_orchestrator,
                        pair_folder=os.path.join(
                            dump_dir, "hole_detection", pair_key
                        ),
                        pair_key=pair_key,
                    )

                if (
                    self.used_conf[ADVANCED][adv_cst.USE_EPIPOLAR_A_PRIORI]
                    is False
                ):
                    # Run epipolar sparse_matching application
                    (
                        pairs[pair_key]["epipolar_matches_left"],
                        _,
                    ) = self.sparse_mtch_app.run(
                        pairs[pair_key]["epipolar_image_left"],
                        pairs[pair_key]["epipolar_image_right"],
                        pairs[pair_key]["grid_left"].attributes[
                            "disp_to_alt_ratio"
                        ],
                        orchestrator=cars_orchestrator,
                        pair_folder=os.path.join(
                            dump_dir, "sparse_matching", pair_key
                        ),
                        pair_key=pair_key,
                    )

                # Run cluster breakpoint to compute sifts: force computation
                cars_orchestrator.breakpoint()

                # Run grid correction application
                save_corrected_grid = (
                    self.epipolar_grid_generation_application.get_save_grids()
                )
                if (
                    self.used_conf[ADVANCED][adv_cst.USE_EPIPOLAR_A_PRIORI]
                    is False
                ):
                    # Estimate grid correction if no epipolar a priori
                    # Filter and save matches
                    pairs[pair_key]["matches_array"] = (
                        self.sparse_mtch_app.filter_matches(
                            pairs[pair_key]["epipolar_matches_left"],
                            pairs[pair_key]["grid_left"],
                            pairs[pair_key]["grid_right"],
                            orchestrator=cars_orchestrator,
                            pair_key=pair_key,
                            pair_folder=os.path.join(
                                dump_dir, "sparse_matching", pair_key
                            ),
                            save_matches=(
                                self.sparse_mtch_app.get_save_matches()
                            ),
                        )
                    )
                    # Compute grid correction
                    (
                        pairs[pair_key]["grid_correction_coef"],
                        pairs[pair_key]["corrected_matches_array"],
                        _,
                        _,
                        _,
                    ) = grid_correction.estimate_right_grid_correction(
                        pairs[pair_key]["matches_array"],
                        pairs[pair_key]["grid_right"],
                        save_matches=self.sparse_mtch_app.get_save_matches(),
                        pair_folder=os.path.join(
                            dump_dir, "grid_correction", "initial", pair_key
                        ),
                        pair_key=pair_key,
                        orchestrator=cars_orchestrator,
                    )
                    # Correct grid right
                    pairs[pair_key]["corrected_grid_right"] = (
                        grid_correction.correct_grid(
                            pairs[pair_key]["grid_right"],
                            pairs[pair_key]["grid_correction_coef"],
                            save_corrected_grid,
                            os.path.join(
                                dump_dir, "grid_correction", "initial", pair_key
                            ),
                        )
                    )

                    # Clean grid at the end of processing if required
                    if not save_corrected_grid:
                        cars_orchestrator.add_to_clean(
                            os.path.join(
                                dump_dir, "grid_correction", "initial", pair_key
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
                        self.sparse_mtch_app.get_matches_filter_knn()
                    )
                    matches_filter_dev_factor = (
                        self.sparse_mtch_app.get_matches_filter_dev_factor()
                    )
                    pairs[pair_key]["filtered_triangulated_matches"] = (
                        sparse_mtch_tools.filter_point_cloud_matches(
                            pairs[pair_key]["triangulated_matches"],
                            matches_filter_knn=matches_filter_knn,
                            matches_filter_dev_factor=matches_filter_dev_factor,
                        )
                    )

                    triangulated_matches_list.append(
                        pairs[pair_key]["filtered_triangulated_matches"]
                    )

            if self.used_conf[ADVANCED][adv_cst.USE_EPIPOLAR_A_PRIORI]:
                # Use a priori
                dem_median = self.used_conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI][
                    adv_cst.DEM_MEDIAN
                ]
                dem_min = self.used_conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI][
                    adv_cst.DEM_MIN
                ]
                dem_max = self.used_conf[ADVANCED][adv_cst.TERRAIN_A_PRIORI][
                    adv_cst.DEM_MAX
                ]

            else:
                dem_generation_output_dir = os.path.join(
                    dump_dir, "dem_generation"
                )
                safe_makedirs(dem_generation_output_dir)
                # Use initial elevation if provided, and generate dems
                # Generate MNT from matches
                dem = self.dem_generation_application.run(
                    triangulated_matches_list,
                    dem_generation_output_dir,
                    self.inputs[sens_cst.INITIAL_ELEVATION][sens_cst.GEOID],
                    dem_roi_to_use=self.dem_generation_roi,
                )
                # Same geometry plugin if we use exogenous dem
                # as initial elevation always used before if provided
                dem_median = dem.attributes[dem_gen_cst.DEM_MEDIAN_PATH]
                if (
                    self.inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]
                    is not None
                ):
                    dem_median = self.inputs[sens_cst.INITIAL_ELEVATION][
                        sens_cst.DEM_PATH
                    ]

                if (
                    dem_median
                    != self.inputs[sens_cst.INITIAL_ELEVATION][
                        sens_cst.DEM_PATH
                    ]
                ):
                    self.geom_plugin_with_dem_and_geoid = (
                        sensor_inputs.generate_geometry_plugin_with_dem(
                            self.used_conf[GEOMETRY_PLUGIN],
                            self.inputs,
                            dem=dem_median,
                            crop_dem=False,
                        )
                    )

                dem_min = dem.attributes[dem_gen_cst.DEM_MIN_PATH]
                dem_max = dem.attributes[dem_gen_cst.DEM_MAX_PATH]

            # update used configuration with terrain a priori
            advanced_parameters.update_conf(
                self.used_conf,
                dem_median=dem_median,
                dem_min=dem_min,
                dem_max=dem_max,
            )

            # Define param
            use_global_disp_range = (
                self.dense_matching_app.use_global_disp_range
            )

            if self.pc_denoising_application is not None:
                denoising_overload_fun = (
                    self.pc_denoising_application.get_triangulation_overload()
                )
            else:
                denoising_overload_fun = None

            pairs_names = [pair_name for pair_name, _, _ in list_sensor_pairs]

            for cloud_id, (pair_key, _, _) in enumerate(list_sensor_pairs):
                # Geometry plugin with dem will be used for the grid generation
                geom_plugin = self.geom_plugin_with_dem_and_geoid
                if (
                    self.used_conf[ADVANCED][adv_cst.USE_EPIPOLAR_A_PRIORI]
                    is False
                ):

                    if not (
                        self.inputs[sens_cst.INITIAL_ELEVATION][
                            sens_cst.DEM_PATH
                        ]
                        is not None
                    ):
                        # Generate grids with new MNT
                        (
                            pairs[pair_key]["new_grid_left"],
                            pairs[pair_key]["new_grid_right"],
                        ) = self.epipolar_grid_generation_application.run(
                            pairs[pair_key]["sensor_image_left"],
                            pairs[pair_key]["sensor_image_right"],
                            geom_plugin,
                            orchestrator=cars_orchestrator,
                            pair_folder=os.path.join(
                                dump_dir,
                                "epipolar_grid_generation",
                                "new_mnt",
                                pair_key,
                            ),
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
                            pairs[pair_key]["corrected_matches_array"],
                            _,
                            _,
                            _,
                        ) = grid_correction.estimate_right_grid_correction(
                            new_grid_matches_array,
                            pairs[pair_key]["new_grid_right"],
                            save_matches=(
                                self.sparse_mtch_app.get_save_matches()
                            ),
                            pair_folder=os.path.join(
                                dump_dir, "grid_correction", "new", pair_key
                            ),
                            pair_key=pair_key,
                            orchestrator=cars_orchestrator,
                        )

                        # Correct grid right

                        pairs[pair_key]["corrected_grid_right"] = (
                            grid_correction.correct_grid(
                                pairs[pair_key]["new_grid_right"],
                                pairs[pair_key]["grid_correction_coef"],
                                save_corrected_grid,
                                os.path.join(
                                    dump_dir, "grid_correction", "new", pair_key
                                ),
                            )
                        )

                        if not save_corrected_grid:
                            cars_orchestrator.add_to_clean(
                                os.path.join(
                                    dump_dir, "grid_correction", "new", pair_key
                                )
                            )

                        # Use the new grid as uncorrected grid
                        pairs[pair_key]["grid_right"] = pairs[pair_key][
                            "new_grid_right"
                        ]

                        pairs[pair_key]["corrected_grid_left"] = pairs[
                            pair_key
                        ]["new_grid_left"]

                    # matches filter params
                    matches_filter_knn = (
                        self.sparse_mtch_app.get_matches_filter_knn()
                    )
                    matches_filter_dev_factor = (
                        self.sparse_mtch_app.get_matches_filter_dev_factor()
                    )
                    if use_global_disp_range:
                        # Triangulate new matches
                        pairs[pair_key]["triangulated_matches"] = (
                            dem_generation_tools.triangulate_sparse_matches(
                                pairs[pair_key]["sensor_image_left"],
                                pairs[pair_key]["sensor_image_right"],
                                pairs[pair_key]["corrected_grid_left"],
                                pairs[pair_key]["corrected_grid_right"],
                                pairs[pair_key]["corrected_matches_array"],
                                geometry_plugin=geom_plugin,
                            )
                        )
                        # filter triangulated_matches
                        # Filter outliers
                        pairs[pair_key]["filtered_triangulated_matches"] = (
                            sparse_mtch_tools.filter_point_cloud_matches(
                                pairs[pair_key]["triangulated_matches"],
                                matches_filter_knn=matches_filter_knn,
                                matches_filter_dev_factor=(
                                    matches_filter_dev_factor
                                ),
                            )
                        )

                    if use_global_disp_range:
                        # Compute disp_min and disp_max
                        (
                            dmin,
                            dmax,
                        ) = sparse_mtch_tools.compute_disp_min_disp_max(
                            pairs[pair_key]["filtered_triangulated_matches"],
                            cars_orchestrator,
                            disp_margin=(
                                self.sparse_mtch_app.get_disparity_margin()
                            ),
                            pair_key=pair_key,
                            disp_to_alt_ratio=pairs[pair_key][
                                "corrected_grid_left"
                            ].attributes["disp_to_alt_ratio"],
                        )
                else:
                    # Use epipolar a priori
                    # load the disparity range
                    [dmin, dmax] = self.used_conf[ADVANCED][
                        adv_cst.EPIPOLAR_A_PRIORI
                    ][pair_key][adv_cst.DISPARITY_RANGE]
                    # load the grid correction coefficient
                    pairs[pair_key]["grid_correction_coef"] = self.used_conf[
                        ADVANCED
                    ][adv_cst.EPIPOLAR_A_PRIORI][pair_key][
                        adv_cst.GRID_CORRECTION
                    ]
                    pairs[pair_key]["corrected_grid_left"] = pairs[pair_key][
                        "grid_left"
                    ]
                    # no correction if the grid correction coefs are None
                    if pairs[pair_key]["grid_correction_coef"] is None:
                        pairs[pair_key]["corrected_grid_right"] = pairs[
                            pair_key
                        ]["grid_right"]
                    else:
                        # Correct grid right with provided epipolar a priori
                        pairs[pair_key]["corrected_grid_right"] = (
                            grid_correction.correct_grid_from_1d(
                                pairs[pair_key]["grid_right"],
                                pairs[pair_key]["grid_correction_coef"],
                                save_corrected_grid,
                                os.path.join(
                                    dump_dir, "grid_correction", pair_key
                                ),
                            )
                        )

                # Run epipolar resampling

                # Update used_conf configuration with epipolar a priori
                # Add global min and max computed with grids
                advanced_parameters.update_conf(
                    self.used_conf,
                    grid_correction_coef=pairs[pair_key][
                        "grid_correction_coef"
                    ],
                    pair_key=pair_key,
                )
                # saved used configuration
                cars_dataset.save_dict(
                    self.used_conf,
                    os.path.join(out_dir, "used_conf.json"),
                    safe_save=True,
                )

                # Generate min and max disp grids
                # Global disparity min and max will be computed from
                # these grids
                dense_matching_pair_folder = os.path.join(
                    dump_dir, "dense_matching", pair_key
                )
                if use_global_disp_range:
                    # Generate min and max disp grids from constants
                    # sensor image is not used here
                    # TODO remove when only local diparity range will be used
                    disp_range_grid = (
                        self.dense_matching_app.generate_disparity_grids(
                            pairs[pair_key]["sensor_image_right"],
                            pairs[pair_key]["corrected_grid_right"],
                            self.geom_plugin_with_dem_and_geoid,
                            dmin=dmin,
                            dmax=dmax,
                            pair_folder=dense_matching_pair_folder,
                        )
                    )
                else:
                    # Generate min and max disp grids from dems
                    disp_range_grid = (
                        self.dense_matching_app.generate_disparity_grids(
                            pairs[pair_key]["sensor_image_right"],
                            pairs[pair_key]["corrected_grid_right"],
                            self.geom_plugin_with_dem_and_geoid,
                            dem_min=dem_min,
                            dem_max=dem_max,
                            dem_median=dem_median,
                            pair_folder=dense_matching_pair_folder,
                        )
                    )

                # Get margins used in dense matching,
                dense_matching_margins_fun = (
                    self.dense_matching_app.get_margins_fun(
                        pairs[pair_key]["corrected_grid_left"], disp_range_grid
                    )
                )

                # TODO add in metadata.json max diff max - min
                # Update used_conf configuration with epipolar a priori
                # Add global min and max computed with grids
                advanced_parameters.update_conf(
                    self.used_conf,
                    dmin=np.min(
                        disp_range_grid[0, 0]["disp_min_grid"].values
                    ),  # TODO compute dmin dans dmax
                    dmax=np.max(disp_range_grid[0, 0]["disp_max_grid"].values),
                    pair_key=pair_key,
                )
                # saved used configuration
                cars_dataset.save_dict(
                    self.used_conf,
                    os.path.join(out_dir, "used_conf.json"),
                    safe_save=True,
                )

                # Generate roi
                epipolar_roi = preprocessing.compute_epipolar_roi(
                    self.input_roi_poly,
                    self.input_roi_epsg,
                    self.geom_plugin_with_dem_and_geoid,
                    pairs[pair_key]["sensor_image_left"],
                    pairs[pair_key]["sensor_image_right"],
                    pairs[pair_key]["corrected_grid_left"],
                    pairs[pair_key]["corrected_grid_right"],
                    os.path.join(dump_dir, "compute_epipolar_roi", pair_key),
                    disp_min=np.min(
                        disp_range_grid[0, 0]["disp_min_grid"].values
                    ),  # TODO compute dmin dans dmax
                    disp_max=np.max(
                        disp_range_grid[0, 0]["disp_max_grid"].values
                    ),
                )

                # Generate new epipolar images
                # Generated with corrected grids
                # Optimal size is computed for the worst case scenario
                # found with epipolar disparity range grids

                (
                    optimum_tile_size,
                    local_tile_optimal_size_fun,
                ) = self.dense_matching_app.get_optimal_tile_size(
                    disp_range_grid,
                    cars_orchestrator.cluster.checked_conf_cluster[
                        "max_ram_per_worker"
                    ],
                )
                (
                    new_epipolar_image_left,
                    new_epipolar_image_right,
                ) = self.resampling_application.run(
                    pairs[pair_key]["sensor_image_left"],
                    pairs[pair_key]["sensor_image_right"],
                    pairs[pair_key]["corrected_grid_left"],
                    pairs[pair_key]["corrected_grid_right"],
                    orchestrator=cars_orchestrator,
                    pair_folder=os.path.join(
                        dump_dir, "resampling", "corrected_grid", pair_key
                    ),
                    pair_key=pair_key,
                    margins_fun=dense_matching_margins_fun,
                    tile_width=optimum_tile_size,
                    tile_height=optimum_tile_size,
                    add_color=True,
                    epipolar_roi=epipolar_roi,
                )

                # Run epipolar matching application
                epipolar_disparity_map = self.dense_matching_app.run(
                    new_epipolar_image_left,
                    new_epipolar_image_right,
                    local_tile_optimal_size_fun,
                    orchestrator=cars_orchestrator,
                    pair_folder=os.path.join(
                        dump_dir, "dense_matching", pair_key
                    ),
                    pair_key=pair_key,
                    disp_range_grid=disp_range_grid,
                    compute_disparity_masks=False,
                    disp_to_alt_ratio=pairs[pair_key][
                        "corrected_grid_left"
                    ].attributes["disp_to_alt_ratio"],
                )

                # Dense matches filling
                if self.dense_matches_filling_1.used_method == "plane":
                    # Fill holes in disparity map
                    (filled_with_1_epipolar_disparity_map) = (
                        self.dense_matches_filling_1.run(
                            epipolar_disparity_map,
                            pairs[pair_key]["holes_bbox_left"],
                            pairs[pair_key]["holes_bbox_right"],
                            disp_min=np.min(
                                disp_range_grid[0, 0]["disp_min_grid"].values
                            ),
                            disp_max=np.max(
                                disp_range_grid[0, 0]["disp_max_grid"].values
                            ),
                            orchestrator=cars_orchestrator,
                            pair_folder=os.path.join(
                                dump_dir, "dense_matches_filling_1", pair_key
                            ),
                            pair_key=pair_key,
                        )
                    )
                else:
                    # Fill with zeros
                    (filled_with_1_epipolar_disparity_map) = (
                        self.dense_matches_filling_1.run(
                            epipolar_disparity_map,
                            orchestrator=cars_orchestrator,
                            pair_folder=os.path.join(
                                dump_dir, "dense_matches_filling_1", pair_key
                            ),
                            pair_key=pair_key,
                        )
                    )

                if self.dense_matches_filling_2.used_method == "plane":
                    # Fill holes in disparity map
                    (filled_with_2_epipolar_disparity_map) = (
                        self.dense_matches_filling_2.run(
                            filled_with_1_epipolar_disparity_map,
                            pairs[pair_key]["holes_bbox_left"],
                            pairs[pair_key]["holes_bbox_right"],
                            disp_min=np.min(
                                disp_range_grid[0, 0]["disp_min_grid"].values
                            ),
                            disp_max=np.max(
                                disp_range_grid[0, 0]["disp_max_grid"].values
                            ),
                            orchestrator=cars_orchestrator,
                            pair_folder=os.path.join(
                                dump_dir, "dense_matches_filling_2", pair_key
                            ),
                            pair_key=pair_key,
                        )
                    )
                else:
                    # Fill with zeros
                    (filled_with_2_epipolar_disparity_map) = (
                        self.dense_matches_filling_2.run(
                            filled_with_1_epipolar_disparity_map,
                            orchestrator=cars_orchestrator,
                            pair_folder=os.path.join(
                                dump_dir, "dense_matches_filling_2", pair_key
                            ),
                            pair_key=pair_key,
                        )
                    )

                if epsg is None:
                    # compute epsg
                    # Epsg uses global disparity min and max
                    epsg = preprocessing.compute_epsg(
                        pairs[pair_key]["sensor_image_left"],
                        pairs[pair_key]["sensor_image_right"],
                        pairs[pair_key]["corrected_grid_left"],
                        pairs[pair_key]["corrected_grid_right"],
                        self.geom_plugin_with_dem_and_geoid,
                        disp_min=np.min(
                            disp_range_grid[0, 0]["disp_min_grid"].values
                        ),
                        disp_max=np.max(
                            disp_range_grid[0, 0]["disp_max_grid"].values
                        ),
                    )
                    # Compute roi polygon, in input EPSG
                    roi_poly = preprocessing.compute_roi_poly(
                        self.input_roi_poly, self.input_roi_epsg, epsg
                    )

                # Checking disparity intervals indicators
                if self.application_conf["dense_matching"][
                    "generate_confidence_intervals"
                ]:
                    intervals = [cst_disp.INTERVAL_INF, cst_disp.INTERVAL_SUP]
                    intervals_pair_flag = False
                    for key, item in self.dense_matching_app.corr_config[
                        "pipeline"
                    ].items():
                        if (
                            cst_disp.CONFIDENCE_KEY in key
                            and item["confidence_method"] == cst_disp.INTERVAL
                        ):
                            indicator = key.split(".")
                            if not intervals_pair_flag:
                                if len(indicator) > 1:
                                    intervals[0] += "." + indicator[-1]
                                    intervals[1] += "." + indicator[-1]
                                # Only processing the first encountered interval
                                intervals_pair_flag = True
                            else:
                                warn_msg = (
                                    "Multiple confidence intervals "
                                    "is not supported. {} will be "
                                    "ignored. Only {} will be processed"
                                ).format(key, intervals)
                                logging.warning(warn_msg)
                else:
                    intervals = None

                if isinstance(self.output[sens_cst.GEOID], str):
                    output_geoid_path = self.output[sens_cst.GEOID]
                elif (
                    isinstance(self.output[sens_cst.GEOID], bool)
                    and self.output[sens_cst.GEOID]
                ):
                    package_path = os.path.dirname(__file__)
                    output_geoid_path = os.path.join(
                        package_path,
                        "..",
                        "..",
                        "conf",
                        sensor_inputs.CARS_GEOID_PATH,
                    )
                else:
                    # default case : stay on the ellipsoid
                    output_geoid_path = None

                depth_map_dir = None
                if self.save_output_depth_map:
                    depth_map_dir = os.path.join(out_dir, "depth_map", pair_key)
                    safe_makedirs(depth_map_dir)

                # Run epipolar triangulation application
                epipolar_points_cloud = self.triangulation_application.run(
                    pairs[pair_key]["sensor_image_left"],
                    pairs[pair_key]["sensor_image_right"],
                    new_epipolar_image_left,
                    pairs[pair_key]["corrected_grid_left"],
                    pairs[pair_key]["corrected_grid_right"],
                    filled_with_2_epipolar_disparity_map,
                    epsg,
                    self.geom_plugin_without_dem_and_geoid,
                    denoising_overload_fun=denoising_overload_fun,
                    source_pc_names=pairs_names,
                    orchestrator=cars_orchestrator,
                    pair_dump_dir=os.path.join(
                        dump_dir, "triangulation", pair_key
                    ),
                    pair_key=pair_key,
                    uncorrected_grid_right=pairs[pair_key]["grid_right"],
                    geoid_path=output_geoid_path,
                    cloud_id=cloud_id,
                    intervals=intervals,
                    pair_output_dir=depth_map_dir,
                    save_output_color=bool(depth_map_dir)
                    and auxiliary[output_constants.AUX_COLOR],
                    save_output_classification=bool(depth_map_dir)
                    and auxiliary[output_constants.AUX_CLASSIFICATION],
                    save_output_filling=bool(depth_map_dir)
                    and auxiliary[output_constants.AUX_FILLING],
                    save_output_mask=bool(depth_map_dir)
                    and auxiliary[output_constants.AUX_MASK],
                    save_output_performance_map=bool(depth_map_dir)
                    and auxiliary[output_constants.AUX_PERFORMANCE_MAP],
                )

                if "no_merging" in self.used_conf[PIPELINE]:
                    denoised_epipolar_points_cloud = (
                        self.pc_denoising_application.run(
                            epipolar_points_cloud,
                            orchestrator=cars_orchestrator,
                            pair_folder=os.path.join(
                                dump_dir, "denoising", pair_key
                            ),
                            pair_key=pair_key,
                        )
                    )

                if self.generate_terrain_products:
                    # Compute terrain bounding box /roi related to
                    # current images
                    (current_terrain_roi_bbox) = (
                        preprocessing.compute_terrain_bbox(
                            pairs[pair_key]["sensor_image_left"],
                            pairs[pair_key]["sensor_image_right"],
                            new_epipolar_image_left,
                            pairs[pair_key]["corrected_grid_left"],
                            pairs[pair_key]["corrected_grid_right"],
                            epsg,
                            self.geom_plugin_with_dem_and_geoid,
                            resolution=resolution,
                            disp_min=np.min(
                                disp_range_grid[0, 0]["disp_min_grid"].values
                            ),
                            disp_max=np.max(
                                disp_range_grid[0, 0]["disp_max_grid"].values
                            ),
                            roi_poly=(
                                None if self.debug_with_roi else roi_poly
                            ),
                            orchestrator=cars_orchestrator,
                            pair_key=pair_key,
                            pair_folder=os.path.join(
                                dump_dir, "terrain_bbox", pair_key
                            ),
                            check_inputs=True,
                        )
                    )
                    list_terrain_roi.append(current_terrain_roi_bbox)

                # add points cloud to list
                if "no_merging" in self.used_conf[PIPELINE]:
                    list_epipolar_points_cloud.append(
                        denoised_epipolar_points_cloud
                    )
                else:
                    list_epipolar_points_cloud.append(epipolar_points_cloud)

            if self.generate_terrain_products:
                # compute terrain bounds
                (
                    terrain_bounds,
                    optimal_terrain_tile_width,
                ) = preprocessing.compute_terrain_bounds(
                    list_terrain_roi,
                    roi_poly=(None if self.debug_with_roi else roi_poly),
                    resolution=resolution,
                )

                if "no_merging" in self.used_conf[PIPELINE]:
                    point_cloud_to_rasterize = (
                        list_epipolar_points_cloud,
                        terrain_bounds,
                    )
                else:
                    # Merge point clouds
                    pc_outliers_removing_1_margins = (
                        self.pc_outliers_removing_1_app.get_on_ground_margin(
                            resolution=resolution
                        )
                    )
                    pc_outliers_removing_2_margins = (
                        self.pc_outliers_removing_2_app.get_on_ground_margin(
                            resolution=resolution
                        )
                    )

                    # find which application produce the final version of the
                    # point cloud. The last generated point cloud will be saved
                    # as official point cloud product if save_output_point_cloud
                    # is True.

                    last_pc_application = None
                    # denoising application will produce a point cloud, unless
                    # it uses the 'none' method.
                    if self.pc_denoising_application.used_method != "none":
                        last_pc_application = "denoising"
                    elif (
                        self.pc_outliers_removing_2_app.used_config.get(
                            "activated", False
                        )
                        is True
                    ):
                        last_pc_application = "pc_outliers_removing_2"
                    elif (
                        self.pc_outliers_removing_1_app.used_config.get(
                            "activated", False
                        )
                        is True
                    ):
                        last_pc_application = "pc_outliers_removing_1"
                    else:
                        last_pc_application = "fusion"

                    merged_points_clouds = self.pc_fusion_application.run(
                        list_epipolar_points_cloud,
                        terrain_bounds,
                        epsg,
                        source_pc_names=pairs_names,
                        orchestrator=cars_orchestrator,
                        margins=(
                            pc_outliers_removing_1_margins
                            + pc_outliers_removing_2_margins
                            + self.rasterization_application.get_margins(
                                resolution
                            )
                        ),
                        optimal_terrain_tile_width=optimal_terrain_tile_width,
                        roi=(roi_poly if self.debug_with_roi else None),
                        save_laz_output=self.save_output_point_cloud
                        and last_pc_application == "fusion",
                    )

                    # Remove outliers with small components method
                    filtered_1_merged_points_clouds = (
                        self.pc_outliers_removing_1_app.run(
                            merged_points_clouds,
                            orchestrator=cars_orchestrator,
                            save_laz_output=self.save_output_point_cloud
                            and last_pc_application == "pc_outliers_removing_1",
                        )
                    )

                    # Remove outliers with statistical components method
                    filtered_2_merged_points_clouds = (
                        self.pc_outliers_removing_2_app.run(
                            filtered_1_merged_points_clouds,
                            orchestrator=cars_orchestrator,
                            save_laz_output=self.save_output_point_cloud
                            and last_pc_application == "pc_outliers_removing_2",
                        )
                    )

                    # denoise point cloud
                    denoised_merged_points_clouds = (
                        self.pc_denoising_application.run(
                            filtered_2_merged_points_clouds,
                            orchestrator=cars_orchestrator,
                            save_laz_output=self.save_output_point_cloud
                            and last_pc_application == "denoising",
                        )
                    )

                    # Rasterize merged and filtered point cloud
                    point_cloud_to_rasterize = denoised_merged_points_clouds

                rasterization_dump_dir = os.path.join(dump_dir, "rasterization")

                dsm_file_name = (
                    os.path.join(
                        out_dir,
                        output_constants.DSM_DIRECTORY,
                        "dsm.tif",
                    )
                    if self.save_output_dsm
                    else None
                )

                color_file_name = (
                    os.path.join(
                        out_dir,
                        output_constants.DSM_DIRECTORY,
                        "color.tif",
                    )
                    if self.save_output_dsm
                    and self.output[output_constants.AUXILIARY][
                        output_constants.AUX_COLOR
                    ]
                    else None
                )

                performance_map_file_name = (
                    os.path.join(
                        out_dir,
                        output_constants.DSM_DIRECTORY,
                        "performance_map.tif",
                    )
                    if self.save_output_dsm
                    and self.output[output_constants.AUXILIARY][
                        output_constants.AUX_PERFORMANCE_MAP
                    ]
                    else None
                )

                classif_file_name = (
                    os.path.join(
                        out_dir,
                        output_constants.DSM_DIRECTORY,
                        "classification.tif",
                    )
                    if self.save_output_dsm
                    and self.output[output_constants.AUXILIARY][
                        output_constants.AUX_CLASSIFICATION
                    ]
                    else None
                )

                mask_file_name = (
                    os.path.join(
                        out_dir,
                        output_constants.DSM_DIRECTORY,
                        "mask.tif",
                    )
                    if self.save_output_dsm
                    and self.output[output_constants.AUXILIARY][
                        output_constants.AUX_MASK
                    ]
                    else None
                )

                contributing_pair_file_name = (
                    os.path.join(
                        out_dir,
                        output_constants.DSM_DIRECTORY,
                        "contributing_pair.tif",
                    )
                    if self.save_output_dsm
                    and self.output[output_constants.AUXILIARY][
                        output_constants.AUX_CONTRIBUTING_PAIR
                    ]
                    else None
                )

                filling_file_name = (
                    os.path.join(
                        out_dir,
                        output_constants.DSM_DIRECTORY,
                        "filling.tif",
                    )
                    if self.save_output_dsm
                    and self.output[output_constants.AUXILIARY][
                        output_constants.AUX_FILLING
                    ]
                    else None
                )

                # rasterize point cloud
                _ = self.rasterization_application.run(
                    point_cloud_to_rasterize,
                    epsg,
                    resolution=resolution,
                    orchestrator=cars_orchestrator,
                    dsm_file_name=dsm_file_name,
                    color_file_name=color_file_name,
                    classif_file_name=classif_file_name,
                    performance_map_file_name=performance_map_file_name,
                    mask_file_name=mask_file_name,
                    contributing_pair_file_name=contributing_pair_file_name,
                    filling_file_name=filling_file_name,
                    color_dtype=list_epipolar_points_cloud[0].attributes[
                        "color_type"
                    ],
                    dump_dir=rasterization_dump_dir,
                )

                # Cleaning: don't keep terrain bbox if save_intermediate_data
                # is not activated
                if not self.advanced[adv_cst.SAVE_INTERMEDIATE_DATA]:
                    cars_orchestrator.add_to_clean(
                        os.path.join(dump_dir, "terrain_bbox")
                    )

            # Cleaning: delete everything in tile_processing if
            # save_intermediate_data is not activated
            if not self.advanced[adv_cst.SAVE_INTERMEDIATE_DATA]:
                cars_orchestrator.add_to_clean(
                    os.path.join(dump_dir, "tile_processing")
                )
