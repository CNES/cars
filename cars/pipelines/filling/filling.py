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
# attribute-defined-outside-init is disabled so that we can create and use
# attributes however we need, to stick to the "everything is attribute" logic
# introduced in issue#895
# pylint: disable=attribute-defined-outside-init
# pylint: disable=too-many-nested-blocks
"""
CARS filling pipeline class file
"""
# pylint: disable=C0302
from __future__ import print_function

import copy
import logging
import os
from collections import OrderedDict

import rasterio
from json_checker import Checker, OptionalKey, Or
from pyproj import CRS

import cars.orchestrator.orchestrator as ocht
from cars.applications.application import Application
from cars.core import cars_logging, inputs, projection, tiling
from cars.core.inputs import read_vector
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset
from cars.orchestrator.cluster.log_wrapper import cars_profile
from cars.pipelines import pipeline_constants as pipeline_cst
from cars.pipelines.filling.filling_pipeline_wrapper import (
    merge_classif_bands_wrapper,
    merge_filling_bands_wrapper,
    monoband_to_multiband_wrapper,
)
from cars.pipelines.parameters import advanced_parameters
from cars.pipelines.parameters import advanced_parameters_constants as adv_cst
from cars.pipelines.parameters import output_constants as out_cst
from cars.pipelines.parameters import output_parameters, sensor_inputs
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from cars.pipelines.parameters.output_constants import AUXILIARY
from cars.pipelines.pipeline import Pipeline
from cars.pipelines.pipeline_constants import (
    ADVANCED,
    APPLICATIONS,
    INPUT,
    ORCHESTRATOR,
    OUTPUT,
)
from cars.pipelines.pipeline_template import PipelineTemplate

PIPELINE = "filling"


@Pipeline.register(
    PIPELINE,
)
class FillingPipeline(PipelineTemplate):
    """
    FillingPipeline
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf, config_dir=None, pre_check=False):  # noqa: C901
        """
        Creates pipeline

        :param pipeline_name: name of the pipeline.
        :type pipeline_name: str
        :param cfg: configuration {'matching_cost_method': value}
        :type cfg: dictionary
        :param config_dir: path to dir containing json or yaml file
        :type config_dir: str
        """

        self.config_dir = config_dir
        # Transform relative path to absolute path
        if config_dir is not None:
            config_dir = os.path.abspath(config_dir)

        # Check global conf
        self.check_global_schema(conf)

        if PIPELINE in conf:
            self.check_pipeline_conf(conf)

        self.out_dir = os.path.abspath(conf[OUTPUT][out_cst.OUT_DIRECTORY])

        self.filling_dir = os.path.join(self.out_dir, "filling")

        # Check input
        conf[INPUT] = self.check_inputs(conf)

        pipeline_conf = conf.get(PIPELINE, {})

        # check advanced
        conf[PIPELINE][ADVANCED] = self.check_advanced(
            pipeline_conf, conf[INPUT]
        )
        # check output
        conf[OUTPUT] = self.check_output(conf)

        self.used_conf = {}

        # Check conf orchestrator
        self.used_conf[ORCHESTRATOR] = self.check_orchestrator(
            conf.get(ORCHESTRATOR, None)
        )
        self.used_conf[INPUT] = conf[INPUT]
        self.used_conf[OUTPUT] = conf[OUTPUT]
        self.used_conf[ADVANCED] = conf[PIPELINE][ADVANCED]
        self.save_all_intermediate_data = self.used_conf[ADVANCED][
            adv_cst.SAVE_INTERMEDIATE_DATA
        ]

        filling_applications = self.generate_filling_applications(
            self.used_conf[INPUT]
        )

        self.filling_tile_size = self.used_conf[ADVANCED][
            adv_cst.FILLING_TILE_SIZE
        ]

        applications_conf = self.overide_pipeline_conf(
            pipeline_conf.get(APPLICATIONS, {}),
            filling_applications,
            append_classification=True,
        )

        self.used_conf[APPLICATIONS] = self.check_applications(
            applications_conf
        )
        # Used classification values, for filling -> will be masked
        self.used_classif_values_for_filling = self.get_classif_values_filling(
            self.used_conf[INPUT]
        )
        self.dump_dir = os.path.join(self.filling_dir, "dump_dir")

        if isinstance(self.used_conf[INPUT][pipeline_cst.DSM_TO_FILL], str):
            self.dsm_to_fill = {
                "dsm": self.used_conf[INPUT][pipeline_cst.DSM_TO_FILL]
            }
        else:
            self.dsm_to_fill = self.used_conf[INPUT][pipeline_cst.DSM_TO_FILL]

        if not pre_check:
            for key, path in self.dsm_to_fill.items():
                self.dsm_to_fill[key] = os.path.abspath(path)

            raster_crs = inputs.rasterio_get_crs(self.dsm_to_fill["dsm"])

            crs = CRS.from_user_input(raster_crs)

            # Un CRS COMPOSÉ contient 2 sous-CRS : horizontal + vertical
            if len(crs.sub_crs_list) == 2:
                self.epsg = crs.sub_crs_list[0].to_epsg()
                self.vertical_crs = inputs.rasterio_get_crs(
                    self.dsm_to_fill["dsm"]
                )

    def check_pipeline_conf(self, conf):
        """
        Check pipeline configuration
        """

        # Validate inputs
        pipeline_schema = {
            OptionalKey(ADVANCED): dict,
            OptionalKey(APPLICATIONS): dict,
        }

        checker_inputs = Checker(pipeline_schema)
        checker_inputs.validate(conf[PIPELINE])

    def check_inputs(self, conf, config_json_dir=None):
        """
        Check the inputs given

        :param conf: configuration
        :type conf: dict
        :param config_dir: directory of used json, if
            user filled paths with relative paths
        :type config_dir: str

        :return: overloader inputs
        :rtype: dict
        """
        return sensor_inputs.sensors_check_inputs(
            conf[INPUT], config_dir=config_json_dir
        )

    def check_output(self, conf):
        """
        Check the output given

        :param conf: configuration of output
        :type conf: dict

        :return overloader output
        :rtype : dict
        """
        conf_output, _ = output_parameters.check_output_parameters(
            conf[INPUT], conf[OUTPUT], self.scaling_coeff
        )
        return conf_output

    def check_advanced(self, conf, conf_input, output_dem_dir=None):
        """
        Check all conf for advanced configuration

        :return: overridden advanced conf
        :rtype: dict
        """

        conf_advanced = conf.get(ADVANCED, {})

        inputs_conf = conf_input

        overloaded_conf = conf_advanced.copy()

        overloaded_conf[adv_cst.SAVE_INTERMEDIATE_DATA] = conf_advanced.get(
            adv_cst.SAVE_INTERMEDIATE_DATA, False
        )
        overloaded_conf[adv_cst.FILLING_TILE_SIZE] = conf_advanced.get(
            adv_cst.FILLING_TILE_SIZE, 10000
        )
        # Check geometry plugin and overwrite geomodel in conf inputs
        (
            inputs_conf,
            overloaded_conf[adv_cst.GEOMETRY_PLUGIN],
            self.geom_plugin_without_dem_and_geoid,
            self.geom_plugin_with_dem_and_geoid,
            self.scaling_coeff,
            _,
        ) = sensor_inputs.check_geometry_plugin(
            inputs_conf,
            conf_advanced.get(adv_cst.GEOMETRY_PLUGIN, None),
            output_dem_dir,
        )

        schema = {
            adv_cst.SAVE_INTERMEDIATE_DATA: Or(dict, bool),
            adv_cst.GEOMETRY_PLUGIN: Or(str, dict),
            adv_cst.FILLING_TILE_SIZE: int,
        }
        checker_advanced_parameters = Checker(schema)
        checker_advanced_parameters.validate(overloaded_conf)

        return overloaded_conf

    def check_applications(self, conf):
        """
        Check the given configuration for applications

        :param conf: configuration of applications
        :type conf: dict
        """
        used_conf = {}
        self.dsm_filling_apps = {}

        needed_applications = []
        needed_applications += [
            "auxiliary_filling",
        ]

        for key in conf:
            if key.startswith("dsm_filling"):
                needed_applications += [key]

        for app_key in needed_applications:
            used_conf[app_key] = conf.get(app_key, {})

            if used_conf[app_key] is None:
                continue
            used_conf[app_key]["save_intermediate_data"] = (
                self.save_all_intermediate_data
                or used_conf[app_key].get("save_intermediate_data", False)
            )

            if app_key == "auxiliary_filling":
                if used_conf[app_key] is not None:
                    used_conf[app_key]["activated"] = used_conf[app_key].get(
                        "activated", True
                    )

        for app_key, app_conf in used_conf.items():
            if not app_key.startswith("dsm_filling"):
                continue

            if app_conf is None:
                self.dsm_filling_apps = {}
                # keep over multiple runs
                used_conf["dsm_filling"] = None
                break

            if app_key in self.dsm_filling_apps:
                msg = (
                    f"The key {app_key} is defined twice in the input "
                    "configuration."
                )
                logging.error(msg)
                raise NameError(msg)

            if app_key[11:] == ".1":
                app_conf.setdefault("method", "exogenous_filling")
            if app_key[11:] == ".2":
                app_conf.setdefault("method", "bulldozer")
            if app_key[11:] == ".3":
                app_conf.setdefault("method", "border_interpolation")

            self.dsm_filling_apps[app_key] = Application(
                "dsm_filling",
                cfg=app_conf,
                scaling_coeff=self.scaling_coeff,
            )
            used_conf[app_key] = self.dsm_filling_apps[app_key].get_conf()

        methods_str = "\n".join(
            f" - {k}={a.used_method}" for k, a in self.dsm_filling_apps.items()
        )
        logging.info(
            "{} dsm filling apps registered:\n{}".format(
                len(self.dsm_filling_apps), methods_str
            )
        )

        # Auxiliary filling
        self.auxiliary_filling_application = Application(
            "auxiliary_filling",
            cfg=conf.get("auxiliary_filling", {}),
            scaling_coeff=self.scaling_coeff,
        )
        used_conf["auxiliary_filling"] = (
            self.auxiliary_filling_application.get_conf()
        )

        # MNT generation
        self.dem_generation_application = Application(
            "dem_generation",
            cfg=used_conf.get("dem_generation", {}),
            scaling_coeff=self.scaling_coeff,
        )
        used_conf["dem_generation"] = self.dem_generation_application.get_conf()

        return used_conf

    def generate_filling_applications(self, inputs_conf):
        """
        Generate filling applications configuration according to inputs

        :param inputs_conf: inputs configuration
        :type inputs_conf: dict
        """

        filling_applications = {}

        # Generate applications configuration
        for filling_name, classif_values in inputs_conf[
            sens_cst.FILLING
        ].items():
            # No filling
            if classif_values is None:
                continue

            classif_values = list(map(str, classif_values))

            # Update application configuration
            if filling_name == "fill_with_geoid":
                new_filling_conf = {
                    "dsm_filling.1": {
                        "method": "exogenous_filling",
                        "classification": classif_values,
                        "fill_with_geoid": classif_values,
                    },
                }
            elif filling_name == "interpolate_from_borders":
                new_filling_conf = {
                    "dsm_filling.2": {
                        "method": "bulldozer",
                        "classification": classif_values,
                    },
                    "dsm_filling.3": {
                        "method": "border_interpolation",
                        "classification": classif_values,
                    },
                }
            elif filling_name == "fill_with_endogenous_dem":
                new_filling_conf = {
                    "dsm_filling.1": {
                        "method": "exogenous_filling",
                        "classification": classif_values,
                    },
                    "dsm_filling.2": {
                        "method": "bulldozer",
                        "classification": classif_values,
                    },
                }
            elif filling_name == "fill_with_exogenous_dem":
                new_filling_conf = {
                    "dsm_filling.2": {
                        "method": "bulldozer",
                        "classification": classif_values,
                    },
                }
            else:
                new_filling_conf = {}

            # Update application configuration
            filling_applications = self.overide_pipeline_conf(
                filling_applications,
                new_filling_conf,
                append_classification=True,
            )

        return filling_applications

    def overide_pipeline_conf(
        self, conf, overiding_conf, append_classification=False
    ):
        """
        Merge two dictionaries recursively without removing keys
        from the base conf.

        :param conf: base configuration dictionary
        :type conf: dict
        :param overiding_conf: overriding configuration dictionary
        :type overiding_conf: dict
        :return: merged configuration
        :rtype: dict
        """
        result = copy.deepcopy(conf)

        def merge_recursive(base_dict, override_dict):
            """
            Main recursive function
            """
            for key, value in override_dict.items():
                if (
                    key in base_dict
                    and isinstance(base_dict[key], dict)
                    and isinstance(value, dict)
                ):
                    merge_recursive(base_dict[key], value)
                elif (
                    append_classification
                    and key in base_dict
                    and isinstance(base_dict[key], list)
                    and isinstance(value, list)
                    and key == "classification"
                ):
                    # extend list, avoiding duplicates
                    base_dict[key] = list(
                        OrderedDict.fromkeys(base_dict[key] + value)
                    )
                else:
                    base_dict[key] = value

        merge_recursive(result, overiding_conf)

        return result

    def get_classif_values_filling(self, inputs_conf):
        """
        Get values in classif, used for filling

        :param inputs_conf: inputs
        :type inputs_conf: dict

        :return: list of values
        :rtype: list
        """

        if (
            sens_cst.FILLING not in inputs_conf
            or inputs_conf[sens_cst.FILLING] is None
        ):
            logging.info("No filling in input configuration")
            return None

        filling_classif_values = []
        for _, classif_values in inputs_conf[sens_cst.FILLING].items():
            # Add new value to filling bands
            if classif_values is not None:
                if isinstance(classif_values, str):
                    classif_values = [classif_values]
                filling_classif_values += classif_values

        simplified_list = list(OrderedDict.fromkeys(filling_classif_values))
        res_as_string_list = [str(value) for value in simplified_list]
        return res_as_string_list

    @cars_profile(name="merge filling bands", interval=0.5)
    def merge_filling_bands(  # pylint: disable=R0917
        self,
        in_filling_path,
        out_filling_path,
        aux_filling,
        dsm_file,
        orchestrator=None,
        tile_size=10000,
    ):
        """
        Merge filling bands to get mono band in output
        """

        if orchestrator is None:
            orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )

        with rasterio.open(in_filling_path) as src:
            nb_bands = src.count

            if nb_bands == 1:
                return None

            profile = src.profile
            height = src.height
            width = src.width
            filling_dtype = src.dtypes[0]
            nodata_value = src.nodata

        # Update to one band
        profile.update(count=1, dtype=filling_dtype)

        filling_cars_ds = cars_dataset.CarsDataset(
            "arrays", name="Monoband Filling"
        )
        # Compute tiling grid
        filling_cars_ds.tiling_grid = tiling.generate_tiling_grid(
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
        ] = orchestrator.get_saving_infos([filling_cars_ds])

        # Save list
        orchestrator.add_to_save_lists(
            out_filling_path,
            "mono_filling",
            filling_cars_ds,
            dtype=filling_dtype,
            nodata=nodata_value,
            optional_data=False,
            cars_ds_name="MonoBand Filling",
        )

        for row in range(filling_cars_ds.shape[0]):
            for col in range(filling_cars_ds.shape[1]):
                # update saving infos  for potential replacement
                full_saving_info = ocht.update_saving_infos(
                    saving_info, row=row, col=col
                )

                window = filling_cars_ds.get_window_as_dict(row, col)
                # Compute images
                (filling_cars_ds[row, col]) = orchestrator.cluster.create_task(
                    merge_filling_bands_wrapper, nout=1
                )(
                    in_filling_path,
                    aux_filling,
                    dsm_file,
                    window=window,
                    saving_info=full_saving_info,
                    profile=profile,
                )

        return filling_cars_ds

    @cars_profile(name="merge classif bands", interval=0.5)
    def merge_classif_bands(  # pylint: disable=R0917
        self,
        in_classif_path,
        out_classif_path,
        aux_classif,
        dsm_file,
        orchestrator=None,
        tile_size=10000,
    ):
        """
        Merge classif bands to get mono band in output
        """
        if orchestrator is None:
            orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )

        with rasterio.open(in_classif_path) as src:
            nb_bands = src.count

            if nb_bands == 1:
                return None

            profile = src.profile
            height = src.height
            width = src.width
            classif_dtype = src.dtypes[0]
            nodata_value = src.nodata

        # Update to one band
        profile.update(count=1, dtype=classif_dtype)

        classif_cars_ds = cars_dataset.CarsDataset(
            "arrays", name="Monoband Classification"
        )
        # Compute tiling grid
        classif_cars_ds.tiling_grid = tiling.generate_tiling_grid(
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
        ] = orchestrator.get_saving_infos([classif_cars_ds])

        # Save list
        orchestrator.add_to_save_lists(
            out_classif_path,
            "classification",
            classif_cars_ds,
            dtype=classif_dtype,
            nodata=nodata_value,
            optional_data=False,
            cars_ds_name="MonoBand Classification",
        )

        for row in range(classif_cars_ds.shape[0]):
            for col in range(classif_cars_ds.shape[1]):
                # update saving infos  for potential replacement
                full_saving_info = ocht.update_saving_infos(
                    saving_info, row=row, col=col
                )

                window = classif_cars_ds.get_window_as_dict(row, col)
                # Compute images
                (classif_cars_ds[row, col]) = orchestrator.cluster.create_task(
                    merge_classif_bands_wrapper, nout=1
                )(
                    in_classif_path,
                    aux_classif,
                    dsm_file,
                    window=window,
                    saving_info=full_saving_info,
                    profile=profile,
                )

        return classif_cars_ds

    def monoband_to_multiband(  # pylint: disable=R0917
        self,
        input_raster,
        output_raster,
        bands_classif,
        orchestrator=None,
        tile_size=10000,
    ):
        """
        Convert classification from monoband to multiband

        :param input_raster: the intput classification path
        :type input_raster: str
        :param output_raster: the output classification path
        :type output_raster: str
        :param bands_classif: the bands values
        :type bands_classif: list
        """

        with rasterio.open(input_raster) as src:
            profile = src.profile
            nodata_value = src.nodata
            height = src.height
            width = src.width
            dtype = src.dtypes[0]

        # update profile
        profile.update(count=len(bands_classif))

        classif_cars_ds = cars_dataset.CarsDataset(
            "arrays", name="Multiband Classification"
        )
        # Compute tiling grid
        classif_cars_ds.tiling_grid = tiling.generate_tiling_grid(
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
        ] = orchestrator.get_saving_infos([classif_cars_ds])

        classif_tags = []
        band_description = []
        for band, classif in enumerate(bands_classif, start=1):
            classif_tags.append((band, classif))
            band_description.append((band, classif))

        # Save list
        orchestrator.add_to_save_lists(
            output_raster,
            "classification",
            classif_cars_ds,
            dtype=dtype,
            nodata=nodata_value,
            optional_data=False,
            rio_tags=classif_tags,
            rio_band_description=band_description,
            cars_ds_name="Multiband Classification",
        )

        for row in range(classif_cars_ds.shape[0]):
            for col in range(classif_cars_ds.shape[1]):

                # update saving infos  for potential replacement
                full_saving_info = ocht.update_saving_infos(
                    saving_info, row=row, col=col
                )

                window = classif_cars_ds.get_window_as_dict(row, col)
                # Compute images
                (classif_cars_ds[row, col]) = orchestrator.cluster.create_task(
                    monoband_to_multiband_wrapper, nout=1
                )(
                    input_raster,
                    bands_classif,
                    nodata_value,
                    window=window,
                    saving_info=full_saving_info,
                    profile=profile,
                )

        return output_raster

    def filling(self):  # noqa: C901
        """
        Filling method
        """
        inputs_conf = self.used_conf[INPUT]

        dsm_file_name = self.dsm_to_fill["dsm"]

        self.texture_bands = self.used_conf[OUTPUT][AUXILIARY][
            out_cst.AUX_IMAGE
        ]

        dsm_filled_dir = os.path.join(self.filling_dir, "dsm/")
        os.makedirs(dsm_filled_dir, exist_ok=True)

        color_file_name = (
            self.dsm_to_fill["image"]
            if "image" in self.used_conf[INPUT][pipeline_cst.DSM_TO_FILL]
            else None
        )

        first_key = list(inputs_conf[sens_cst.SENSORS].keys())[0]
        input_classif = inputs_conf[sens_cst.SENSORS][first_key][
            sens_cst.INPUT_CLASSIFICATION
        ]
        bands_classif = None
        if input_classif is not None:
            bands_classif = input_classif["values"]

        classif_file_name = None
        if sens_cst.INPUT_CLASSIFICATION:
            classif_file_name = os.path.join(
                dsm_filled_dir, "classification.tif"
            )
            self.monoband_to_multiband(
                self.dsm_to_fill["classification"],
                classif_file_name,
                bands_classif,
                orchestrator=self.cars_orchestrator,
                tile_size=self.filling_tile_size,
            )
            self.cars_orchestrator.breakpoint()

        filling_file_name = (
            self.dsm_to_fill["filling"]
            if "filling" in self.used_conf[INPUT][pipeline_cst.DSM_TO_FILL]
            else None
        )

        if inputs_conf[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH] is None:
            dems = {}

            dem_generation_output_dir = os.path.join(
                self.dump_dir, "dem_generation"
            )
            safe_makedirs(dem_generation_output_dir)

            # Use initial elevation if provided, and generate dems
            _, paths, _ = self.dem_generation_application.run(
                dsm_file_name,
                dem_generation_output_dir,
                input_geoid=self.used_conf[INPUT][sens_cst.INITIAL_ELEVATION][
                    sens_cst.GEOID
                ],
                output_geoid=self.used_conf[OUTPUT][out_cst.OUT_GEOID],
                initial_elevation=(
                    self.used_conf[INPUT][sens_cst.INITIAL_ELEVATION][
                        sens_cst.DEM_PATH
                    ]
                ),
                default_alt=self.geom_plugin_with_dem_and_geoid.default_alt,
                cars_orchestrator=self.cars_orchestrator,
            )

            dem_median = paths["dem_median"]
            dem_min = paths["dem_min"]
            dem_max = paths["dem_max"]

            dems["dem_median"] = dem_median
            dems["dem_min"] = dem_min
            dems["dem_max"] = dem_max

            # Override initial elevation
            inputs_conf[sens_cst.INITIAL_ELEVATION][
                sens_cst.DEM_PATH
            ] = dem_median

            # Check advanced parameters with new initial elevation
            output_dem_dir = os.path.join(
                self.used_conf[OUTPUT][out_cst.OUT_DIRECTORY],
                "dump_dir",
                "initial_elevation",
            )
            safe_makedirs(output_dem_dir)
            (
                inputs_conf,
                _,
                self.geometry_plugin,
                self.geom_plugin_without_dem_and_geoid,
                self.geom_plugin_with_dem_and_geoid,
                _,
                _,
                _,
                _,
            ) = advanced_parameters.check_advanced_parameters(
                inputs_conf,
                self.used_conf.get(ADVANCED, {}),
                output_dem_dir=output_dem_dir,
            )

        if not hasattr(self, "list_intersection_poly"):
            if (
                self.used_conf[INPUT][sens_cst.INITIAL_ELEVATION][
                    sens_cst.DEM_PATH
                ]
                is not None
            ):
                sensor_inputs.load_geomodels(
                    inputs_conf, self.geom_plugin_without_dem_and_geoid
                )
                self.list_sensor_pairs = sensor_inputs.generate_pairs(
                    self.used_conf[INPUT]
                )

                self.list_intersection_poly = []
                for _, (
                    pair_key,
                    sensor_image_left,
                    sensor_image_right,
                ) in enumerate(self.list_sensor_pairs):
                    pair_folder = os.path.join(
                        self.dump_dir, "terrain_bbox", pair_key
                    )
                    safe_makedirs(pair_folder)
                    geojson1 = os.path.join(
                        pair_folder, "left_envelope.geojson"
                    )
                    geojson2 = os.path.join(
                        pair_folder, "right_envelope.geojson"
                    )
                    out_envelopes_intersection = os.path.join(
                        pair_folder, "envelopes_intersection.geojson"
                    )

                    inter_poly, _ = projection.ground_intersection_envelopes(
                        sensor_image_left[sens_cst.INPUT_IMG]["bands"]["b0"][
                            "path"
                        ],
                        sensor_image_right[sens_cst.INPUT_IMG]["bands"]["b0"][
                            "path"
                        ],
                        sensor_image_left[sens_cst.INPUT_GEO_MODEL],
                        sensor_image_right[sens_cst.INPUT_GEO_MODEL],
                        self.geom_plugin_with_dem_and_geoid,
                        geojson1,
                        geojson2,
                        out_envelopes_intersection,
                        envelope_file_driver="GeoJSON",
                        intersect_file_driver="GeoJSON",
                    )

                    # Retrieve bounding box of the grd inters of the envelopes
                    inter_poly, inter_epsg = read_vector(
                        out_envelopes_intersection
                    )

                    # Project polygon if epsg is different
                    if self.vertical_crs != CRS(inter_epsg):
                        inter_poly = projection.polygon_projection_crs(
                            inter_poly, CRS(inter_epsg), self.vertical_crs
                        )

                self.list_intersection_poly.append(inter_poly)
            else:
                self.list_intersection_poly = None

        dtm_file_name = None
        for app_key, app in self.dsm_filling_apps.items():

            app_dump_dir = os.path.join(
                self.dump_dir, app_key.replace(".", "_")
            )

            if app.get_conf()["method"] == "exogenous_filling":
                _ = app.run(
                    dsm_file=dsm_file_name,
                    classif_file=classif_file_name,
                    filling_file=filling_file_name,
                    dump_dir=app_dump_dir,
                    roi_polys=self.list_intersection_poly,
                    roi_epsg=self.epsg,
                    output_geoid=self.used_conf[OUTPUT][sens_cst.GEOID],
                    geom_plugin=self.geom_plugin_with_dem_and_geoid,
                    dsm_dir=dsm_filled_dir,
                    tile_size=self.filling_tile_size,
                    orchestrator=self.cars_orchestrator,
                )
            elif app.get_conf()["method"] == "bulldozer":
                _, dtm_file_name = app.run(
                    dsm_file=dsm_file_name,
                    classif_file=classif_file_name,
                    filling_file=filling_file_name,
                    dump_dir=app_dump_dir,
                    roi_polys=self.list_intersection_poly,
                    roi_epsg=self.epsg,
                    orchestrator=self.cars_orchestrator,
                    dsm_dir=dsm_filled_dir,
                    tile_size=self.filling_tile_size,
                )
            elif app.get_conf()["method"] == "border_interpolation":
                _ = app.run(
                    dsm_file=dsm_file_name,
                    classif_file=classif_file_name,
                    filling_file=filling_file_name,
                    dtm_file=dtm_file_name,
                    dump_dir=app_dump_dir,
                    roi_polys=self.list_intersection_poly,
                    roi_epsg=self.epsg,
                    dsm_dir=dsm_filled_dir,
                    orchestrator=self.cars_orchestrator,
                    tile_size=self.filling_tile_size,
                )

            if not app.save_intermediate_data:
                self.cars_orchestrator.add_to_clean(app_dump_dir)

            if dsm_file_name == self.dsm_to_fill["dsm"]:
                dsm_file_name = os.path.join(dsm_filled_dir, "dsm.tif")

            if "filling" in self.used_conf[INPUT][pipeline_cst.DSM_TO_FILL]:
                if filling_file_name == self.dsm_to_fill["filling"]:
                    filling_file_name = os.path.join(
                        dsm_filled_dir, "filling.tif"
                    )

            self.cars_orchestrator.breakpoint()

        _ = self.auxiliary_filling_application.run(
            dsm_file=os.path.join(dsm_filled_dir, "dsm.tif"),
            color_file=color_file_name,
            classif_file=classif_file_name,
            dump_dir=self.dump_dir,
            sensor_inputs=self.used_conf[INPUT].get("sensors"),
            pairing=self.used_conf[INPUT].get("pairing"),
            geom_plugin=self.geom_plugin_with_dem_and_geoid,
            texture_bands=self.texture_bands,
            output_geoid=self.used_conf[OUTPUT][sens_cst.GEOID],
            orchestrator=self.cars_orchestrator,
            dsm_dir=dsm_filled_dir,
        )
        self.cars_orchestrator.breakpoint()

        if (
            os.path.join(dsm_filled_dir, "classification.tif") is not None
            and self.used_conf[OUTPUT][out_cst.AUXILIARY][
                out_cst.AUX_CLASSIFICATION
            ]
        ):
            tmp_classif_file_name = os.path.join(
                dsm_filled_dir, "tmp_classification.tif"
            )
            merged_classif = self.merge_classif_bands(
                classif_file_name,
                tmp_classif_file_name,
                self.used_conf[OUTPUT][out_cst.AUXILIARY][
                    out_cst.AUX_CLASSIFICATION
                ],
                dsm_file_name,
                orchestrator=self.cars_orchestrator,
                tile_size=self.filling_tile_size,
            )
            self.cars_orchestrator.breakpoint()
            # move tmp classif to final classif
            if merged_classif is not None:
                os.replace(tmp_classif_file_name, classif_file_name)
        if (
            filling_file_name is not None
            and self.used_conf[OUTPUT][out_cst.AUXILIARY][out_cst.AUX_FILLING]
        ):
            tmp_filling_file_name = os.path.join(
                dsm_filled_dir, "tmp_filling.tif"
            )

            merged_filling = self.merge_filling_bands(
                os.path.join(dsm_filled_dir, "filling.tif"),
                tmp_filling_file_name,
                self.used_conf[OUTPUT][out_cst.AUXILIARY][out_cst.AUX_FILLING],
                dsm_file_name,
                orchestrator=self.cars_orchestrator,
            )
            self.cars_orchestrator.breakpoint()
            # move tmp fillinf to final filling
            if merged_filling is not None:
                os.replace(
                    tmp_filling_file_name,
                    os.path.join(dsm_filled_dir, "filling.tif"),
                )

        return True

    @cars_profile(name="Run_filling_pipeline", interval=0.5)
    def run(self, args=None, log_dir=None):  # noqa C901 # pylint: disable=W0613
        """
        Run pipeline
        """
        cars_logging.add_progress_message("Starting filling pipeline")

        self.log_dir = os.path.join(self.filling_dir, "logs")
        if log_dir is not None:
            self.log_dir = log_dir
        elif (
            args is not None
            and hasattr(args, "log_dir")
            and args.log_dir is not None
        ):
            self.log_dir = args.log_dir

        with ocht.Orchestrator(
            orchestrator_conf=self.used_conf[ORCHESTRATOR],
            out_dir=self.filling_dir,
            log_dir=self.log_dir,
            out_yaml_path=os.path.join(
                self.filling_dir,
                out_cst.INFO_FILENAME,
            ),
        ) as self.cars_orchestrator:
            self.filling()
