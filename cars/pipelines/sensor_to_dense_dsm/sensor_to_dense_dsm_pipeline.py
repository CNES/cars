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
CARS sensors_to_dense_dsm pipeline class file
"""
# pylint: disable=too-many-lines
# Standard imports
from __future__ import print_function

import json
import logging
import os

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
from cars.core import cars_logging, preprocessing, roi_tools
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.core.inputs import get_descriptions_bands
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset
from cars.orchestrator import orchestrator
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


@Pipeline.register("sensors_to_dense_dsm", "sensors_to_dense_point_clouds")
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
        self.used_conf[PIPELINE] = conf.get(PIPELINE, "sensors_to_dense_dsm")

        self.generate_terrain_products = True
        # set json pipeline file
        if self.used_conf[PIPELINE] == "sensors_to_dense_dsm":
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

        # Check geometry plugin and overwrite geomodel in conf inputs
        (
            self.inputs,
            self.used_conf[GEOMETRY_PLUGIN],
            self.geom_plugin_without_dem_and_geoid,
            self.geom_plugin_with_dem_and_geoid,
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

        self.debug_with_roi = self.used_conf[INPUTS][sens_cst.DEBUG_WITH_ROI]

        # Check conf output
        self.output = self.check_output(self.conf[OUTPUT])
        self.used_conf[OUTPUT] = self.output

        # Check conf application
        self.application_conf = self.check_applications(
            self.conf.get(APPLICATIONS, {}), self.generate_terrain_products
        )
        self.used_conf[APPLICATIONS] = self.application_conf

        # Check conf application vs inputs application
        self.check_inputs_with_applications(self.inputs, self.application_conf)

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
        return sensors_inputs.sensors_check_inputs(
            conf, config_json_dir=config_json_dir
        )

    @staticmethod
    def check_output(conf):
        """
        Check the output given

        :param conf: configuration of output
        :type conf: dict

        :return overloader output
        :rtype : dict
        """
        return dsm_output.dense_dsm_check_output(conf)

    def check_applications(self, conf, generate_terrain_products):
        """
        Check the given configuration for applications,
        and generates needed applications for pipeline.

        :param conf: configuration of applications
        :type conf: dict
        :param generate_terrain_products: true if uses point cloud
            fusion, pc removing, rasterization
        :type generate_terrain_products: bool
        """

        # Check if all specified applications are used
        # Application in terrain_application are note used in
        # the sensors_to_dense_point_clouds pipeline
        needed_applications = [
            "grid_generation",
            "resampling",
            "holes_detection",
            "dense_matches_filling.1",
            "dense_matches_filling.2",
            "sparse_matching",
            "dense_matching",
            "triangulation",
            "dem_generation",
        ]

        terrain_applications = [
            "point_cloud_fusion",
            "point_cloud_rasterization",
            "point_cloud_outliers_removing.1",
            "point_cloud_outliers_removing.2",
        ]

        pipeline_name = "sensors_to_dense_point_clouds"
        if generate_terrain_products:
            needed_applications += terrain_applications
            pipeline_name = "sensors_to_dense_dsm"

        # Initialize used config
        used_conf = {}
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

        # Epipolar grid generation
        self.epipolar_grid_generation_application = Application(
            "grid_generation", cfg=conf.get("grid_generation", {})
        )
        used_conf[
            "grid_generation"
        ] = self.epipolar_grid_generation_application.get_conf()

        # image resampling
        self.resampling_application = Application(
            "resampling", cfg=conf.get("resampling", {})
        )
        used_conf["resampling"] = self.resampling_application.get_conf()

        # holes detection
        self.holes_detection_app = Application(
            "holes_detection", cfg=conf.get("holes_detection", {})
        )
        used_conf["holes_detection"] = self.holes_detection_app.get_conf()

        # disparity filling 1 plane
        self.dense_matches_filling_1 = Application(
            "dense_matches_filling",
            cfg=conf.get(
                "dense_matches_filling.1",
                {"method": "plane"},
            ),
        )
        used_conf[
            "dense_matches_filling.1"
        ] = self.dense_matches_filling_1.get_conf()

        # disparity filling  2
        self.dense_matches_filling_2 = Application(
            "dense_matches_filling",
            cfg=conf.get(
                "dense_matches_filling.2",
                {"method": "zero_padding"},
            ),
        )
        used_conf[
            "dense_matches_filling.2"
        ] = self.dense_matches_filling_2.get_conf()

        # Sparse Matching
        self.sparse_mtch_app = Application(
            "sparse_matching", cfg=conf.get("sparse_matching", {})
        )
        used_conf["sparse_matching"] = self.sparse_mtch_app.get_conf()

        # Matching
        self.dense_matching_application = Application(
            "dense_matching", cfg=conf.get("dense_matching", {})
        )
        used_conf["dense_matching"] = self.dense_matching_application.get_conf()

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

        if generate_terrain_products:
            # Points cloud fusion
            self.pc_fusion_application = Application(
                "point_cloud_fusion", cfg=conf.get("point_cloud_fusion", {})
            )
            used_conf[
                "point_cloud_fusion"
            ] = self.pc_fusion_application.get_conf()

            # Points cloud outlier removing small components
            self.pc_outliers_removing_1_app = Application(
                "point_cloud_outliers_removing",
                cfg=conf.get(
                    "point_cloud_outliers_removing.1",
                    {"method": "small_components"},
                ),
            )
            used_conf[
                "point_cloud_outliers_removing.1"
            ] = self.pc_outliers_removing_1_app.get_conf()

            # Points cloud outlier removing statistical
            self.pc_outliers_removing_2_app = Application(
                "point_cloud_outliers_removing",
                cfg=conf.get(
                    "point_cloud_outliers_removing.2",
                    {"method": "statistical"},
                ),
            )
            used_conf[
                "point_cloud_outliers_removing.2"
            ] = self.pc_outliers_removing_2_app.get_conf()

            # Rasterization
            self.rasterization_application = Application(
                "point_cloud_rasterization",
                cfg=conf.get("point_cloud_rasterization", {}),
            )
            used_conf[
                "point_cloud_rasterization"
            ] = self.rasterization_application.get_conf()
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

    @staticmethod
    def check_inputs_with_applications(inputs_conf, application_conf):
        """
        Check for each application the input configuration consistency

        :param inputs_conf: inputs checked configuration
        :type inputs_conf: dict
        :param application_conf: application checked configuration
        :type application_conf: dict
        """

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

    def run(self):  # noqa C901
        """
        Run pipeline

        """

        out_dir = self.output["out_dir"]
        cars_logging.add_log_file(out_dir, "sensor_to_dense_dsm")

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
                    "pipeline": "sensor_to_dense_dsm_pipeline",
                    "inputs": self.inputs,
                }
            )

            # Run applications

            # Initialize epsg for terrain tiles
            epsg = self.inputs[sens_cst.EPSG]
            if epsg is not None:
                # Compute roi polygon, in input EPSG
                roi_poly = preprocessing.compute_roi_poly(
                    self.input_roi_poly, self.input_roi_epsg, epsg
                )

            # List of terrain roi corresponding to each epipolar pair
            # Used to generate final terrain roi
            list_terrain_roi = []

            # initialise lists of points
            list_epipolar_points_cloud = []
            list_sensor_pairs = sensors_inputs.generate_inputs(
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
                # Create Pair folder
                pair_folder = os.path.join(out_dir, pair_key)
                safe_makedirs(pair_folder)
                tmp_dir = os.path.join(pair_folder, "tmp")
                safe_makedirs(tmp_dir)
                cars_orchestrator.add_to_clean(tmp_dir)

                # initialize pairs for current pair
                pairs[pair_key] = {}
                pairs[pair_key]["pair_folder"] = pair_folder
                pairs[pair_key]["sensor_image_left"] = sensor_image_left
                pairs[pair_key]["sensor_image_right"] = sensor_image_right

                # Run applications

                # Run grid generation
                # We generate grids with dem if it is provided.
                # If not provided, grid are generated without dem and a dem
                # will be generated, to use later for a new grid generation
                if self.inputs[sens_cst.INITIAL_ELEVATION] is None:
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
                    pair_folder=pairs[pair_key]["pair_folder"],
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

                if self.used_conf[INPUTS]["use_epipolar_a_priori"] is False or (
                    len(pairs[pair_key]["holes_classif"]) > 0
                ):
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
                        pair_folder=pairs[pair_key]["pair_folder"],
                        pair_key=pair_key,
                        margins=self.sparse_mtch_app.get_margins(),
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
                        pair_folder=pairs[pair_key]["pair_folder"],
                        pair_key=pair_key,
                    )

                if self.used_conf[INPUTS]["use_epipolar_a_priori"] is False:
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
                        pair_folder=pairs[pair_key]["pair_folder"],
                        pair_key=pair_key,
                    )

                # Run cluster breakpoint to compute sifts: force computation
                cars_orchestrator.breakpoint()

                # Run grid correction application
                save_corrected_grid = (
                    self.epipolar_grid_generation_application.save_grids
                )
                if self.used_conf[INPUTS]["use_epipolar_a_priori"] is False:
                    # Estimate grid correction if no epipolar a priori
                    # Filter matches
                    matches_array = self.sparse_mtch_app.filter_matches(
                        pairs[pair_key]["epipolar_matches_left"],
                        orchestrator=cars_orchestrator,
                        pair_key=pair_key,
                        pair_folder=pairs[pair_key]["pair_folder"],
                        save_matches=self.sparse_mtch_app.get_save_matches(),
                    )

                    # Compute grid correction
                    (
                        pairs[pair_key]["grid_correction_coef"],
                        pairs[pair_key]["corrected_matches_array"],
                        _,
                        _,
                        _,
                    ) = grid_correction.estimate_right_grid_correction(
                        matches_array,
                        pairs[pair_key]["grid_right"],
                        save_matches=self.sparse_mtch_app.get_save_matches(),
                        pair_folder=pairs[pair_key]["pair_folder"],
                        pair_key=pair_key,
                        orchestrator=cars_orchestrator,
                    )

                    # Correct grid right
                    pairs[pair_key][
                        "corrected_grid_right"
                    ] = grid_correction.correct_grid(
                        pairs[pair_key]["grid_right"],
                        pairs[pair_key]["grid_correction_coef"],
                        save_corrected_grid,
                        pairs[pair_key]["pair_folder"],
                    )

                    pairs[pair_key]["corrected_grid_left"] = pairs[pair_key][
                        "grid_left"
                    ]

                    # Triangulate matches
                    pairs[pair_key][
                        "triangulated_matches"
                    ] = dem_generation_tools.triangulate_sparse_matches(
                        pairs[pair_key]["sensor_image_left"],
                        pairs[pair_key]["sensor_image_right"],
                        pairs[pair_key]["grid_left"],
                        pairs[pair_key]["corrected_grid_right"],
                        pairs[pair_key]["corrected_matches_array"],
                        self.geom_plugin_with_dem_and_geoid,
                    )
                    triangulated_matches_list.append(
                        pairs[pair_key]["triangulated_matches"]
                    )

                    # Clean variables
                    del matches_array

            # For now only dem_mean will be used and is mandatory for
            # a priory
            dem_mean = self.inputs[sens_cst.INITIAL_ELEVATION]
            dem_min = None
            dem_max = None

            if self.inputs[sens_cst.INITIAL_ELEVATION] is None and (
                self.used_conf[INPUTS]["use_epipolar_a_priori"] is False
            ):
                # Generate MNT from matches
                dem = self.dem_generation_application.run(
                    triangulated_matches_list, cars_orchestrator.out_dir
                )

                # Generate geometry loader with dem and geoid
                self.geom_plugin_with_dem_and_geoid = (
                    sensors_inputs.generate_geometry_plugin_with_dem(
                        self.used_conf[GEOMETRY_PLUGIN],
                        self.inputs,
                        dem=dem.attributes[dem_gen_cst.DEM_MEAN_PATH],
                    )
                )
                dem_mean = dem.attributes[dem_gen_cst.DEM_MEAN_PATH]
                dem_min = dem.attributes[dem_gen_cst.DEM_MIN_PATH]
                dem_max = dem.attributes[dem_gen_cst.DEM_MAX_PATH]

            # update used configuration with terrain a priori
            sensors_inputs.update_conf(
                self.used_conf,
                dem_mean=dem_mean,
                dem_min=dem_min,
                dem_max=dem_max,
            )

            for pair_key, _, _ in list_sensor_pairs:
                # Geometry plugin with dem will be used for the grid generation
                geom_plugin = self.geom_plugin_with_dem_and_geoid

                if self.used_conf[INPUTS]["use_epipolar_a_priori"] is False:
                    if self.inputs[sens_cst.INITIAL_ELEVATION] is None:
                        # Generate grids with new MNT
                        (
                            pairs[pair_key]["new_grid_left"],
                            pairs[pair_key]["new_grid_right"],
                        ) = self.epipolar_grid_generation_application.run(
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
                            pair_folder=pairs[pair_key]["pair_folder"],
                            pair_key=pair_key,
                            orchestrator=cars_orchestrator,
                        )

                        # Correct grid right
                        pairs[pair_key][
                            "corrected_grid_right"
                        ] = grid_correction.correct_grid(
                            pairs[pair_key]["new_grid_right"],
                            pairs[pair_key]["grid_correction_coef"],
                        )
                        pairs[pair_key]["corrected_grid_left"] = pairs[
                            pair_key
                        ]["new_grid_left"]

                        # Triangulate new matches
                        pairs[pair_key][
                            "triangulated_matches"
                        ] = dem_generation_tools.triangulate_sparse_matches(
                            pairs[pair_key]["sensor_image_left"],
                            pairs[pair_key]["sensor_image_right"],
                            pairs[pair_key]["corrected_grid_left"],
                            pairs[pair_key]["corrected_grid_right"],
                            pairs[pair_key]["corrected_matches_array"],
                            geometry_plugin=geom_plugin,
                        )

                    # Compute disp_min and disp_max
                    (dmin, dmax) = sparse_mtch_tools.compute_disp_min_disp_max(
                        pairs[pair_key]["triangulated_matches"],
                        cars_orchestrator,
                        disp_margin=(
                            self.sparse_mtch_app.get_disparity_margin()
                        ),
                        pair_key=pair_key,
                        disp_to_alt_ratio=pairs[pair_key][
                            "corrected_grid_left"
                        ].attributes["disp_to_alt_ratio"],
                        matches_filter_knn=(
                            self.sparse_mtch_app.get_matches_filter_knn()
                        ),
                        matches_filter_std_factor=(
                            self.sparse_mtch_app.get_matches_filter_std_factor()
                        ),
                    )

                    # Clean variables
                    del pairs[pair_key]["corrected_matches_array"]
                    del pairs[pair_key]["epipolar_matches_left"]
                else:
                    # Use epipolar a priori
                    # load the disparity range
                    [dmin, dmax] = self.used_conf[INPUTS]["epipolar_a_priori"][
                        pair_key
                    ]["disparity_range"]
                    # load the grid correction coefficient
                    pairs[pair_key]["grid_correction_coef"] = self.used_conf[
                        INPUTS
                    ]["epipolar_a_priori"][pair_key]["grid_correction"]
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
                        pairs[pair_key][
                            "corrected_grid_right"
                        ] = grid_correction.correct_grid_from_1d(
                            pairs[pair_key]["grid_right"],
                            pairs[pair_key]["grid_correction_coef"],
                            save_corrected_grid,
                            pair_folder,
                        )

                # Update used_conf configuration with epipolar a priori
                sensors_inputs.update_conf(
                    self.used_conf,
                    grid_correction_coef=pairs[pair_key][
                        "grid_correction_coef"
                    ],
                    dmin=dmin,
                    dmax=dmax,
                    pair_key=pair_key,
                )
                # saved used configuration
                cars_dataset.save_dict(
                    self.used_conf,
                    os.path.join(out_dir, "used_conf.json"),
                    safe_save=True,
                )
                # Run epipolar resampling

                # Get margins used in dense matching,
                # with updated disp min and max
                (
                    dense_matching_margins,
                    disp_min,
                    disp_max,
                ) = self.dense_matching_application.get_margins(
                    pairs[pair_key]["corrected_grid_left"],
                    disp_min=dmin,
                    disp_max=dmax,
                )

                # if sequential mode, apply roi
                epipolar_roi = None
                if (
                    cars_orchestrator.cluster.checked_conf_cluster["mode"]
                    == "sequential"
                ):
                    # Generate roi
                    epipolar_roi = preprocessing.compute_epipolar_roi(
                        self.input_roi_poly,
                        self.input_roi_epsg,
                        self.geom_plugin_with_dem_and_geoid,
                        pairs[pair_key]["sensor_image_left"],
                        pairs[pair_key]["sensor_image_right"],
                        pairs[pair_key]["corrected_grid_left"],
                        pairs[pair_key]["corrected_grid_right"],
                        pairs[pair_key]["pair_folder"],
                        disp_min=disp_min,
                        disp_max=disp_max,
                    )

                # Generate new epipolar images
                # Generated with corrected grids
                (
                    new_epipolar_image_left,
                    new_epipolar_image_right,
                ) = self.resampling_application.run(
                    pairs[pair_key]["sensor_image_left"],
                    pairs[pair_key]["sensor_image_right"],
                    pairs[pair_key]["corrected_grid_left"],
                    pairs[pair_key]["corrected_grid_right"],
                    orchestrator=cars_orchestrator,
                    pair_folder=pairs[pair_key]["pair_folder"],
                    pair_key=pair_key,
                    margins=dense_matching_margins,
                    optimum_tile_size=(
                        self.dense_matching_application.get_optimal_tile_size(
                            disp_min,
                            disp_max,
                            cars_orchestrator.cluster.checked_conf_cluster[
                                "max_ram_per_worker"
                            ],
                        )
                    ),
                    add_color=True,
                    epipolar_roi=epipolar_roi,
                )

                # Run epipolar matching application
                epipolar_disparity_map = self.dense_matching_application.run(
                    new_epipolar_image_left,
                    new_epipolar_image_right,
                    orchestrator=cars_orchestrator,
                    pair_folder=pairs[pair_key]["pair_folder"],
                    pair_key=pair_key,
                    disp_min=disp_min,
                    disp_max=disp_max,
                    compute_disparity_masks=False,
                    disp_to_alt_ratio=pairs[pair_key][
                        "corrected_grid_left"
                    ].attributes["disp_to_alt_ratio"],
                )

                # Dense matches filling
                if self.dense_matches_filling_1.used_method == "plane":
                    # Fill holes in disparity map
                    (
                        filled_with_1_epipolar_disparity_map
                    ) = self.dense_matches_filling_1.run(
                        epipolar_disparity_map,
                        pairs[pair_key]["holes_bbox_left"],
                        pairs[pair_key]["holes_bbox_right"],
                        disp_min=disp_min,
                        disp_max=disp_max,
                        orchestrator=cars_orchestrator,
                        pair_folder=pairs[pair_key]["pair_folder"],
                        pair_key=pair_key,
                    )
                else:
                    # Fill with zeros
                    (
                        filled_with_1_epipolar_disparity_map
                    ) = self.dense_matches_filling_1.run(
                        epipolar_disparity_map,
                        orchestrator=cars_orchestrator,
                        pair_folder=pairs[pair_key]["pair_folder"],
                        pair_key=pair_key,
                    )

                if self.dense_matches_filling_2.used_method == "plane":
                    # Fill holes in disparity map
                    (
                        filled_with_2_epipolar_disparity_map
                    ) = self.dense_matches_filling_2.run(
                        filled_with_1_epipolar_disparity_map,
                        pairs[pair_key]["holes_bbox_left"],
                        pairs[pair_key]["holes_bbox_right"],
                        disp_min=disp_min,
                        disp_max=disp_max,
                        orchestrator=cars_orchestrator,
                        pair_folder=pairs[pair_key]["pair_folder"],
                        pair_key=pair_key,
                    )
                else:
                    # Fill with zeros
                    (
                        filled_with_2_epipolar_disparity_map
                    ) = self.dense_matches_filling_2.run(
                        filled_with_1_epipolar_disparity_map,
                        orchestrator=cars_orchestrator,
                        pair_folder=pairs[pair_key]["pair_folder"],
                        pair_key=pair_key,
                    )

                if epsg is None:
                    # compute epsg
                    epsg = preprocessing.compute_epsg(
                        pairs[pair_key]["sensor_image_left"],
                        pairs[pair_key]["sensor_image_right"],
                        pairs[pair_key]["corrected_grid_left"],
                        pairs[pair_key]["corrected_grid_right"],
                        self.geom_plugin_with_dem_and_geoid,
                        orchestrator=cars_orchestrator,
                        pair_folder=pairs[pair_key]["pair_folder"],
                        disp_min=disp_min,
                        disp_max=disp_max,
                    )
                    # Compute roi polygon, in input EPSG
                    roi_poly = preprocessing.compute_roi_poly(
                        self.input_roi_poly, self.input_roi_epsg, epsg
                    )

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
                    orchestrator=cars_orchestrator,
                    pair_folder=pairs[pair_key]["pair_folder"],
                    pair_key=pair_key,
                    uncorrected_grid_right=pairs[pair_key]["grid_right"],
                    geoid_path=self.inputs[sens_cst.GEOID],
                    disp_min=disp_min,
                    disp_max=disp_max,
                )

                if self.generate_terrain_products:
                    # Compute terrain bounding box /roi related to
                    # current images
                    (
                        current_terrain_roi_bbox
                    ) = preprocessing.compute_terrain_bbox(
                        pairs[pair_key]["sensor_image_left"],
                        pairs[pair_key]["sensor_image_right"],
                        new_epipolar_image_left,
                        pairs[pair_key]["corrected_grid_left"],
                        pairs[pair_key]["corrected_grid_right"],
                        epsg,
                        self.geom_plugin_with_dem_and_geoid,
                        resolution=(
                            self.rasterization_application.get_resolution()
                        ),
                        disp_min=disp_min,
                        disp_max=disp_max,
                        roi_poly=(None if self.debug_with_roi else roi_poly),
                        orchestrator=cars_orchestrator,
                        pair_key=pair_key,
                        pair_folder=pairs[pair_key]["pair_folder"],
                        check_inputs=self.inputs[sens_cst.CHECK_INPUTS],
                    )
                    list_terrain_roi.append(current_terrain_roi_bbox)

                # add points cloud to list
                list_epipolar_points_cloud.append(epipolar_points_cloud)

            if self.generate_terrain_products:
                # compute terrain bounds
                (
                    terrain_bounds,
                    optimal_terrain_tile_width,
                ) = preprocessing.compute_terrain_bounds(
                    list_terrain_roi,
                    roi_poly=(None if self.debug_with_roi else roi_poly),
                    resolution=self.rasterization_application.get_resolution(),
                )

                # Merge point clouds
                merged_points_clouds = self.pc_fusion_application.run(
                    list_epipolar_points_cloud,
                    terrain_bounds,
                    epsg,
                    orchestrator=cars_orchestrator,
                    margins=(
                        self.pc_outliers_removing_1_app.get_on_ground_margin(
                            resolution=(
                                self.rasterization_application.get_resolution()
                            )
                        )
                        + self.pc_outliers_removing_2_app.get_on_ground_margin(
                            resolution=(
                                self.rasterization_application.get_resolution()
                            )
                        )
                        + self.rasterization_application.get_margins()
                    ),
                    optimal_terrain_tile_width=optimal_terrain_tile_width,
                    roi=(roi_poly if self.debug_with_roi else None),
                )

                # Add pair names to retrieve source pair of each point
                pairs_names = [
                    pair_name for pair_name, _, _ in list_sensor_pairs
                ]
                merged_points_clouds.attributes["source_pc_names"] = pairs_names

                # Remove outliers with small components method
                filtered_1_merged_points_clouds = (
                    self.pc_outliers_removing_1_app.run(
                        merged_points_clouds,
                        orchestrator=cars_orchestrator,
                    )
                )

                # Remove outliers with statistical components method
                filtered_2_merged_points_clouds = (
                    self.pc_outliers_removing_2_app.run(
                        filtered_1_merged_points_clouds,
                        orchestrator=cars_orchestrator,
                    )
                )

                # rasterize point cloud
                _ = self.rasterization_application.run(
                    filtered_2_merged_points_clouds,
                    epsg,
                    orchestrator=cars_orchestrator,
                    dsm_file_name=os.path.join(
                        out_dir, self.output[sens_cst.DSM_BASENAME]
                    ),
                    color_file_name=os.path.join(
                        out_dir, self.output[sens_cst.CLR_BASENAME]
                    ),
                )
