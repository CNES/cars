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
CARS analysis pipeline class file
"""
import copy
import json
import logging
import math
import os

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping

from cars import extractroi
from cars.core import projection
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.core.utils import safe_makedirs
from cars.pipelines.analysis.report_tools import (
    generate_report_cars_output,
    generate_satelite_position_report,
    merge_reports,
)
from cars.pipelines.default.default_pipeline import DefaultPipeline
from cars.pipelines.parameters import advanced_parameters_constants as adv_cst
from cars.pipelines.parameters import output_constants as out_cst
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from cars.pipelines.pipeline import Pipeline
from cars.pipelines.pipeline_constants import (
    ADVANCED,
    INPUT,
    OUTPUT,
    SUBSAMPLING,
    TIE_POINTS,
)
from cars.pipelines.pipeline_template import PipelineTemplate
from cars.pipelines.subsampling.subsampling import SubsamplingPipeline
from cars.pipelines.surface_modeling.surface_modeling import (
    SurfaceModelingPipeline,
)

PIPELINE = "analysis"


@Pipeline.register(
    PIPELINE,
)
class AnalysisPipeline(PipelineTemplate):
    """
    Analysis pipeline
    """

    def __init__(self, conf, config_dir=None):
        """
        Creates pipeline

        Directly creates class attributes:
            used_conf

        :param pipeline_name: name of the pipeline.
        :type pipeline_name: str
        :param cfg: configuration {'matching_cost_method': value}
        :type cfg: dictionary
        :param config_dir: path to dir containing json/yaml
        :type config_dir: str
        """

        self.used_conf = {"input": conf["input"], "output": conf["output"]}
        self.config_dir = config_dir

    @staticmethod
    def check_inputs(conf, config_dir=None):  # pylint: disable=unused-argument
        """
        Check the inputs given

        :param conf: configuration of inputs
        :type conf: dict
        :param config_dir: directory of used json/yaml, if
            user filled paths with relative paths
        :type config_dir: str

        :return: overloaded inputs
        :rtype: dict
        """

        return conf

    @staticmethod
    def check_output(conf):
        """
        Check the output given

        :param conf: configuration of output
        :type conf: dict
        :return: overloader output
        :rtype: dict
        """

        return conf

    def run(self, args=None, log_dir=None):  # pylint: disable=W0613
        """
        Run pipeline

        """

        if not self.used_conf[INPUT][sens_cst.SENSORS]:
            raise ValueError("No sensor provided in input")

        pipeline_out_dir = self.used_conf[OUTPUT][out_cst.OUT_DIRECTORY]
        intermediate_data_dir = os.path.join(
            pipeline_out_dir, "intermediate_data"
        )

        # Generate overview of input images

        # Generate terain envelope of input images

        # Run satelite position report
        satelite_positions, sat_infos = generate_satelite_position(
            copy.deepcopy(self.used_conf), self.config_dir
        )
        sat_data_dir = os.path.join(intermediate_data_dir, "sat_data")
        sat_report = generate_satelite_position_report(
            satelite_positions, sat_infos, sat_data_dir
        )

        # Run Low res pipeline : dsm
        low_res_dir = os.path.join(intermediate_data_dir, "low_res")
        low_res_report, low_res_error, used_resolution = (
            launch_low_res_pipeline(
                self.used_conf, low_res_dir, self.config_dir
            )
        )

        # Run full res pipeline on crop : tie point analysis, and dsm
        full_res_dir = os.path.join(intermediate_data_dir, "full_res")
        full_res_report, full_res_error = launch_full_res_pipeline(
            copy.deepcopy(self.used_conf),
            full_res_dir,
            low_res_dir,
            self.config_dir,
            used_resolution,
        )

        # Generate report
        reports_to_merge = [sat_report, low_res_report, full_res_report]
        merged_report_pdf = os.path.join(pipeline_out_dir, "report.pdf")
        merged_report_html = os.path.join(pipeline_out_dir, "report.html")
        merge_reports(reports_to_merge, merged_report_html, merged_report_pdf)

        if low_res_error or full_res_error:
            raise RuntimeError(
                "CARS pipeline did not run successfully, "
                f"check report for more details in {merged_report_html}"
            )


def launch_low_res_pipeline(conf, low_res_dir, config_dir):
    """
    Launch full res pipeline on crop : tie point analysis, and dsm
    """

    safe_makedirs(low_res_dir)
    report_file = os.path.join(low_res_dir, "report.html")

    # launch cars
    log_error = "No error during low res pipeline"
    error = False
    try:

        # launch cars on low res images
        low_res_conf = copy.deepcopy(conf)
        if sens_cst.ROI in low_res_conf[INPUT][sens_cst.SENSORS]:
            del low_res_conf[INPUT][sens_cst.SENSORS][sens_cst.ROI]
        if sens_cst.FILLING in low_res_conf[INPUT]:
            del low_res_conf[INPUT][sens_cst.FILLING]

        # compute resolution to use
        subsampling_dir = os.path.join(low_res_dir, "tmp_subsampling")
        safe_makedirs(low_res_dir)
        subsampling_conf = {
            INPUT: conf[INPUT],
            OUTPUT: {
                out_cst.OUT_DIRECTORY: subsampling_dir,
            },
        }
        subsampling_pipeline = SubsamplingPipeline(
            subsampling_conf, config_dir=config_dir
        )

        subsamp_advanced = subsampling_pipeline.check_advanced(
            {
                "resolutions": [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
                "min_image_size": 300,
            },
            subsampling_conf[INPUT],
        )
        # Get resolution: higher one
        resolution = subsamp_advanced[adv_cst.RESOLUTIONS][0]

        cars_conf = {
            INPUT: low_res_conf[INPUT],
            OUTPUT: {
                out_cst.OUT_DIRECTORY: low_res_dir,
                out_cst.AUXILIARY: {
                    out_cst.AUX_IMAGE: True,
                    out_cst.AUX_AMBIGUITY: True,
                },
            },
            SUBSAMPLING: {"advanced": {"resolutions": [resolution]}},
            TIE_POINTS: {ADVANCED: {"all": {"save_intermediate_data": True}}},
        }

        saved_cars_conf = copy.deepcopy(cars_conf)

        default_pipeline = DefaultPipeline(cars_conf, config_dir=config_dir)
        default_pipeline.run()

    except Exception as exc:
        # Generate report with error message
        log_error = "Error during low res pipeline: {}".format(exc)
        logging.error(log_error)
        error = True

    # Generate report
    generate_report_cars_output(
        report_file, low_res_dir, log_error, saved_cars_conf
    )

    return report_file, error, resolution


def launch_full_res_pipeline(
    conf, full_res_dir, low_res_dir, config_dir, used_resolution
):
    """
    Launch full res pipeline on crop : tie point analysis, and dsm
    """
    if not os.path.exists(full_res_dir):
        os.makedirs(full_res_dir)

    # retrieve best roi in 4326
    roi = get_best_roi(
        low_res_dir, tile_size=int(math.ceil(2000 / used_resolution))
    )

    # Generate pivot format
    if "pipeline" in conf:
        del conf["pipeline"]
    if "loaders" not in conf["input"]:
        conf["input"]["loaders"] = {}

    # fix bug: image is in pivot format
    conf["input"]["loaders"]["image"] = "pivot"

    default_pipeline_dry = DefaultPipeline(conf, config_dir=config_dir)
    input_conf = default_pipeline_dry.used_conf[0]["input"]
    # crop images on best roi
    cropped_images_dir = os.path.join(full_res_dir, "cropped_images")
    if not os.path.exists(cropped_images_dir):
        os.makedirs(cropped_images_dir)
    new_conf_input = generate_cropped_images(
        input_conf, cropped_images_dir, roi
    )

    # launch cars on crop
    log_error = "No error during full res pipeline"
    error = False
    try:
        cars_conf = {
            INPUT: new_conf_input,
            OUTPUT: {
                out_cst.OUT_DIRECTORY: full_res_dir,
                out_cst.AUXILIARY: {
                    out_cst.AUX_IMAGE: True,
                    out_cst.AUX_AMBIGUITY: True,
                },
            },
            TIE_POINTS: {ADVANCED: {"all": {"save_intermediate_data": True}}},
        }

        saved_cars_conf = copy.deepcopy(cars_conf)

        default_pipeline = DefaultPipeline(cars_conf, config_dir=config_dir)
        default_pipeline.run()

    except Exception as exc:
        # Generate report with error message
        log_error = "Error during full res pipeline: {}".format(exc)
        logging.error(log_error)
        error = True

    # Generate report on crop results

    report_file = os.path.join(full_res_dir, "report.html")
    # Generate report
    generate_report_cars_output(
        report_file, full_res_dir, log_error, saved_cars_conf
    )

    return report_file, error


def generate_cropped_images(  # noqa: C901
    input_conf, cropped_images_dir, roi_bbox
):
    """
    Generate cropped image and associated conf
    :param input_conf:
    :param cropped_images_dir:
    :param roi_bbox:
    :return:
    """

    # Get pivot format conf
    new_input_conf = copy.deepcopy(input_conf)

    # Update dem
    rpc_options = {}
    dem = new_input_conf["initial_elevation"]["dem"]
    if dem is not None:
        rpc_options["rpc_dem"] = dem

    # new_input_conf in in pivot format
    for sensor_key, sensor in new_input_conf[sens_cst.SENSORS].items():
        # Get geomodel
        if sensor["geomodel"]["model_type"] != "RPC":
            raise ValueError("Only RPC geomodel is supported for now")
        geomodel = sensor["geomodel"]["path"]
        # get all images
        images = []
        for _, band_dict in sensor["image"]["bands"].items():
            if band_dict["path"] not in images:
                images.append(band_dict["path"])

        mask = sensor.get("mask", None)

        classification = sensor.get("classification", None)
        if classification is not None:
            classification = classification["path"]

        if mask is not None:
            images.append(mask)
        if classification is not None:
            images.append(classification)

        # crop images
        for img in images:
            # crop
            cropped_image = os.path.join(
                cropped_images_dir, sensor_key + "_" + os.path.basename(img)
            )

            extractroi.process_image_file(
                roi_bbox,
                img,
                cropped_image,
                None,
                rpc_options,
                external_rpc_file=geomodel,
            )

            # replace in conf
            for _, band_dict in sensor["image"]["bands"].items():
                if band_dict["path"] == img:
                    band_dict["path"] = cropped_image

            if mask is not None:
                if sensor["mask"] == img:
                    sensor["mask"] = cropped_image

            if classification is not None:
                if sensor["classification"]["path"] == img:
                    sensor["classification"]["path"] = cropped_image

    return new_input_conf


def get_best_roi(low_res_dir, tile_size=2000):
    """
    Get best roi from low res results
    """

    dsm_file = os.path.join(low_res_dir, "dsm", "dsm.tif")
    ambiguity_file = os.path.join(low_res_dir, "dsm", "ambiguity.tif")

    with (
        rasterio.open(dsm_file) as dsm_src,
        rasterio.open(ambiguity_file) as ambi_src,
    ):
        nodata = dsm_src.nodata
        crs = dsm_src.crs
        best_score = -1
        # best_tile_coords :  (minx, miny, maxx, maxy)
        best_tile_coords = None

        # Generate tiling grid
        for col in range(0, dsm_src.width, tile_size):
            for row in range(0, dsm_src.height, tile_size):
                window = rasterio.windows.Window(
                    col,
                    row,
                    min(tile_size, dsm_src.width - col),
                    min(tile_size, dsm_src.height - row),
                )

                window_bounds = rasterio.windows.bounds(
                    window, dsm_src.transform
                )

                # Read data for this tile
                dsm_data = dsm_src.read(1, window=window)
                ambi_data = ambi_src.read(1, window=window)

                # Calculate metrics: Valid pixels ratio and mean confidence
                valid_mask = (
                    (dsm_data != nodata)
                    if nodata is not None
                    else np.isfinite(dsm_data)
                )
                valid_count = np.count_nonzero(valid_mask)

                if valid_count > 0:
                    mean_ambi = np.mean(ambi_data[valid_mask])
                    # Score: priority to coverage, then quality
                    score = valid_count * (1 - mean_ambi)

                    if score > best_score:
                        best_score = score
                        best_tile_coords = window_bounds

        if not best_tile_coords:
            return None

        # Reproject to EPSG:4326
        # (west, south, east, north)
        bbox_roi = transform_bounds(crs, "EPSG:4326", *best_tile_coords)

        geom = box(*bbox_roi)

        # Create GeoJSON structure
        geojson_feature = {
            "type": "Feature",
            "properties": {"name": "ROI Selection", "buffer_margin": "10%"},
            "geometry": mapping(geom),
        }

        geosjon = json.dumps(geojson_feature, indent=2)
        logging.info(f"Best ROI GeoJSON:\n{geosjon}")

        return bbox_roi


def generate_satelite_position(cars_conf, config_dir):
    """
    Generate satellite positions
    :param cars_conf:
    :param config_dir:
    :return:
    """
    cars_conf = {
        "input": cars_conf["input"],
        "output": cars_conf["output"],
    }

    surface_modeling_pipeline = SurfaceModelingPipeline(
        cars_conf, config_dir=config_dir
    )
    full_cars_conf = surface_modeling_pipeline.used_conf

    geom_plugin = AbstractGeometry("SharelocGeometry")  # pylint: disable=E0110

    sensors = full_cars_conf["input"]["sensors"]
    skeys = list(sensors.keys())
    sat_positions = []

    infos = {"convergence_angles": {}}

    for idx, skey in enumerate(skeys[1:]):
        angles = projection.get_ground_angles(
            sensors[skeys[0]]["image"]["bands"]["b0"]["path"],
            sensors[skey]["image"]["bands"]["b0"]["path"],
            sensors[skeys[0]]["geomodel"],
            sensors[skey]["geomodel"],
            geom_plugin,
        )
        azl, ell, azr, elr, conv = angles
        if len(sat_positions) == 0:
            sat_positions.append(["0", azl, ell])
        sat_positions.append([str(idx + 1), azr, elr])

        conv_key = f"0 <-> {idx + 1} , {skeys[0]} <-> {skey}"
        infos["convergence_angles"][conv_key] = conv

    return sat_positions, infos
