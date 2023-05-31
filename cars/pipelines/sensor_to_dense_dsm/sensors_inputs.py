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
CARS containing inputs checking for sensor input data
Used for full_res and low_res pipelines
"""

import logging
import os

import rasterio as rio
from json_checker import Checker, Or

# CARS imports
from cars.core import inputs
from cars.core.utils import make_relative_path_absolute
from cars.pipelines.sensor_to_dense_dsm import (
    sensor_dense_dsm_constants as sens_cst,
)

CARS_GEOID_PATH = "geoid/egm96.grd"  # Path in cars package (pkg)


def sensors_check_inputs(  # noqa: C901
    conf, config_json_dir=None, check_epipolar_a_priori=True
):
    """
    Check the inputs given

    :param conf: configuration of inputs
    :type conf: dict
    :param config_json_dir: path to dir containing json
    :type config_json_dir: str
    """

    overloaded_conf = conf.copy()

    # Overload some optional parameters
    overloaded_conf[sens_cst.EPSG] = conf.get(sens_cst.EPSG, None)
    overloaded_conf[sens_cst.INITIAL_ELEVATION] = conf.get(
        sens_cst.INITIAL_ELEVATION, None
    )
    overloaded_conf[sens_cst.DEFAULT_ALT] = conf.get(sens_cst.DEFAULT_ALT, 0)
    overloaded_conf[sens_cst.ROI] = conf.get(sens_cst.ROI, None)
    overloaded_conf[sens_cst.CHECK_INPUTS] = conf.get(
        sens_cst.CHECK_INPUTS, False
    )

    if check_epipolar_a_priori:
        # Check conf use_epipolar_a_priori
        overloaded_conf["use_epipolar_a_priori"] = conf.get(
            "use_epipolar_a_priori", False
        )
        # Retrieve epipolar_a_priori if it is provided
        if "epipolar_a_priori" in conf:
            overloaded_conf["epipolar_a_priori"] = conf.get(
                "epipolar_a_priori", {}
            )
        else:
            overloaded_conf["epipolar_a_priori"] = {}

    if "geoid" not in overloaded_conf:
        # use cars geoid
        logging.info("CARS will use its own internal file as geoid reference")
        # Get root package directory
        package_path = os.path.dirname(__file__)
        geoid_path = os.path.join(
            package_path, "..", "..", "conf", CARS_GEOID_PATH
        )
        overloaded_conf[sens_cst.GEOID] = geoid_path
    else:
        overloaded_conf[sens_cst.GEOID] = conf.get(sens_cst.GEOID, None)

    # Validate inputs
    inputs_schema = {
        sens_cst.SENSORS: dict,
        sens_cst.PAIRING: [[str]],
        sens_cst.EPSG: Or(int, None),  # move to rasterization
        sens_cst.INITIAL_ELEVATION: Or(str, None),
        sens_cst.DEFAULT_ALT: int,
        sens_cst.ROI: Or(str, dict, None),
        sens_cst.CHECK_INPUTS: bool,
        sens_cst.GEOID: Or(None, str),
    }
    if check_epipolar_a_priori:
        inputs_schema[sens_cst.USE_EPIPOLAR_A_PRIORI] = bool
        inputs_schema[sens_cst.EPIPOLAR_A_PRIORI] = dict

    checker_inputs = Checker(inputs_schema)
    checker_inputs.validate(overloaded_conf)

    # Validate epipolar schema
    epipolar_schema = {
        sens_cst.GRID_CORRECTION: Or(list, None),
        sens_cst.DISPARITY_RANGE: list,
    }

    checker_epipolar = Checker(epipolar_schema)

    # Validate each sensor image
    sensor_schema = {
        sens_cst.INPUT_IMG: str,
        sens_cst.INPUT_COLOR: str,
        sens_cst.INPUT_NODATA: int,
        sens_cst.INPUT_GEO_MODEL: str,
        sens_cst.INPUT_GEO_MODEL_TYPE: str,
        sens_cst.INPUT_GEO_MODEL_FILTER: Or([str], None),
        sens_cst.INPUT_MSK: Or(str, None),
        sens_cst.INPUT_CLASSIFICATION: Or(str, None),
    }
    checker_sensor = Checker(sensor_schema)

    for sensor_image_key in conf[sens_cst.SENSORS]:
        # Overload optional parameters
        color = conf[sens_cst.SENSORS][sensor_image_key].get(
            "color",
            conf[sens_cst.SENSORS][sensor_image_key][sens_cst.INPUT_IMG],
        )
        overloaded_conf[sens_cst.SENSORS][sensor_image_key][
            sens_cst.INPUT_COLOR
        ] = color

        geomodel_type = conf[sens_cst.SENSORS][sensor_image_key].get(
            sens_cst.INPUT_GEO_MODEL_TYPE, "RPC"
        )
        overloaded_conf[sens_cst.SENSORS][sensor_image_key][
            sens_cst.INPUT_GEO_MODEL_TYPE
        ] = geomodel_type

        geomodel_filters = conf[sens_cst.SENSORS][sensor_image_key].get(
            sens_cst.INPUT_GEO_MODEL_FILTER, None
        )
        overloaded_conf[sens_cst.SENSORS][sensor_image_key][
            sens_cst.INPUT_GEO_MODEL_FILTER
        ] = geomodel_filters

        no_data = conf[sens_cst.SENSORS][sensor_image_key].get(
            sens_cst.INPUT_NODATA, -9999
        )
        overloaded_conf[sens_cst.SENSORS][sensor_image_key][
            sens_cst.INPUT_NODATA
        ] = no_data

        mask = conf[sens_cst.SENSORS][sensor_image_key].get(
            sens_cst.INPUT_MSK, None
        )
        overloaded_conf[sens_cst.SENSORS][sensor_image_key][
            sens_cst.INPUT_MSK
        ] = mask
        classif = conf[sens_cst.SENSORS][sensor_image_key].get(
            sens_cst.INPUT_CLASSIFICATION, None
        )
        overloaded_conf[sens_cst.SENSORS][sensor_image_key][
            sens_cst.INPUT_CLASSIFICATION
        ] = classif
        # Validate
        checker_sensor.validate(
            overloaded_conf[sens_cst.SENSORS][sensor_image_key]
        )

    # check epipolar a priori for each image pair
    if (
        check_epipolar_a_priori
        and overloaded_conf[sens_cst.USE_EPIPOLAR_A_PRIORI]
    ):
        validate_epipolar_a_priori(conf, overloaded_conf, checker_epipolar)

    # Validate pairs
    for key1, key2 in overloaded_conf[sens_cst.PAIRING]:
        if key1 not in overloaded_conf[sens_cst.SENSORS]:
            logging.error("{} not in sensors images".format(key1))
            raise RuntimeError("{} not in sensors images".format(key1))
        if key2 not in overloaded_conf["sensors"]:
            logging.error("{} not in sensors images".format(key2))
            raise RuntimeError("{} not in sensors images".format(key2))

    # Modify to absolute path
    if config_json_dir is not None:
        modify_to_absolute_path(config_json_dir, overloaded_conf)

    else:
        logging.debug(
            "path of config file was not given,"
            "relative path are not transformed to absolute paths"
        )

    if not overloaded_conf[sens_cst.CHECK_INPUTS]:
        logging.info(
            "The inputs consistency will not be checked. "
            "To enable the inputs checking, add check_inputs: True "
            "to your input configuration"
        )

    # Check image, msk and color size compatibility
    for sensor_image_key in overloaded_conf[sens_cst.SENSORS]:
        sensor_image = overloaded_conf[sens_cst.SENSORS][sensor_image_key]
        check_input_size(
            sensor_image[sens_cst.INPUT_IMG],
            sensor_image[sens_cst.INPUT_MSK],
            sensor_image[sens_cst.INPUT_COLOR],
            sensor_image[sens_cst.INPUT_CLASSIFICATION],
        )
        # check band nbits of msk and classification
        check_nbits(
            sensor_image[sens_cst.INPUT_MSK],
            sensor_image[sens_cst.INPUT_CLASSIFICATION],
        )
        # check image and color data consistency
        if overloaded_conf[sens_cst.CHECK_INPUTS]:
            check_input_data(
                sensor_image[sens_cst.INPUT_IMG],
                sensor_image[sens_cst.INPUT_COLOR],
            )

    # Check srtm dir
    check_srtm(overloaded_conf[sens_cst.INITIAL_ELEVATION])

    return overloaded_conf


def modify_to_absolute_path(config_json_dir, overloaded_conf):
    """
    Modify input file path to absolute path

    :param config_json_dir: directory of the json configuration
    :type config_json_dir: str
    :param overloaded_conf: overloaded configuration json
    :dict overloaded_conf: dict
    """
    for sensor_image_key in overloaded_conf[sens_cst.SENSORS]:
        sensor_image = overloaded_conf[sens_cst.SENSORS][sensor_image_key]
        for tag in [
            sens_cst.INPUT_IMG,
            sens_cst.INPUT_MSK,
            sens_cst.INPUT_GEO_MODEL,
            sens_cst.INPUT_COLOR,
            sens_cst.INPUT_CLASSIFICATION,
        ]:
            if sensor_image[tag] is not None:
                sensor_image[tag] = make_relative_path_absolute(
                    sensor_image[tag], config_json_dir
                )

    for tag in [sens_cst.INITIAL_ELEVATION, sens_cst.ROI, sens_cst.GEOID]:
        if overloaded_conf[tag] is not None:
            if isinstance(overloaded_conf[tag], str):
                overloaded_conf[tag] = make_relative_path_absolute(
                    overloaded_conf[tag], config_json_dir
                )


def validate_epipolar_a_priori(conf, overloaded_conf, checker_epipolar):
    """
    Validate inner epipolar configuration

    :param conf : input configuration json
    :type conf: dict
    :param overloaded_conf : overloaded configuration json
    :type overloaded_conf: dict
    :param checker_epipolar : json checker
    :type checker_epipolar: Checker
    """

    for key_image_pair in conf[sens_cst.EPIPOLAR_A_PRIORI]:
        checker_epipolar.validate(
            overloaded_conf[sens_cst.EPIPOLAR_A_PRIORI][key_image_pair]
        )


def check_srtm(srtm_dir):
    """
    Check srtm data

    :param srtm_dir: directory of srtm
    :type srtm_dir: str
    """

    if srtm_dir is not None:
        if os.path.isdir(srtm_dir):
            srtm_tiles = os.listdir(srtm_dir)
            if len(srtm_tiles) == 0:
                logging.warning(
                    "SRTM directory is empty, "
                    "the default altitude will be used as reference altitude."
                )
            else:
                logging.info(
                    "Indicated SRTM tiles valid regions "
                    "will be used as reference altitudes "
                    "(the default altitude is used "
                    "for undefined regions of the SRTM)"
                )
        else:
            # TODO add check for single file
            pass
    else:
        logging.info("The default altitude will be used as reference altitude.")


def check_input_data(image, color):
    """
    Check data of the image and color

    :param image: image path
    :type image: str
    :param color: color path
    :type color: str
    """

    with rio.open(image) as img_reader:
        trans = img_reader.transform
        if trans.e < 0:
            logging.warning(
                "{} seems to have an incoherent pixel size. "
                "Input images has to be in sensor geometry.".format(image)
            )

    with rio.open(color) as img_reader:
        trans = img_reader.transform
        if trans.e < 0:
            logging.warning(
                "{} seems to have an incoherent pixel size. "
                "Input images has to be in sensor geometry.".format(image)
            )


def check_input_size(image, mask, color, classif):
    """
    Check image, mask, classif and color given

    Images must have same size

    :param image: image path
    :type image: str
    :param mask: mask path
    :type mask: str
    :param color: color path
    :type color: str
    :param classif: classif path
    :type classif: str
    """
    if inputs.rasterio_get_nb_bands(image) != 1:
        raise RuntimeError("{} is not mono-band images".format(image))

    if mask is not None:
        if inputs.rasterio_get_size(image) != inputs.rasterio_get_size(mask):
            raise RuntimeError(
                "The image {} and the mask {} "
                "do not have the same size".format(image, mask)
            )

    if color is not None:
        if inputs.rasterio_get_size(image) != inputs.rasterio_get_size(color):
            raise RuntimeError(
                "The image {} and the color {} "
                "do not have the same size".format(image, color)
            )

    if classif is not None:
        if inputs.rasterio_get_size(image) != inputs.rasterio_get_size(classif):
            raise RuntimeError(
                "The image {} and the classif {} "
                "do not have the same size".format(image, classif)
            )


def check_nbits(mask, classif):
    """
    Check the bits number of the mask, classif
    mask and classification are limited to 1 bits per band

    :param mask: mask path
    :type mask: str
    :param classif: classif path
    :type classif: str
    """

    if mask is not None:
        nbits = inputs.rasterio_get_nbits(mask)
        if not check_all_nbits_equal_one(nbits):
            raise RuntimeError(
                "The mask {} have {} nbits per band. ".format(mask, nbits)
                + "Only the mask with nbits=1 is supported! "
            )

    if classif is not None:
        nbits = inputs.rasterio_get_nbits(classif)
        if not check_all_nbits_equal_one(nbits):
            raise RuntimeError(
                "The classification {} have {} nbits per band. ".format(
                    classif, nbits
                )
                + "Only the classification with nbits=1 is supported! "
            )


def check_all_nbits_equal_one(nbits):
    """
    Check if all the nbits = 1
    :param nbits: list of the nbits
    :return: True if all the nbits = 1
    """
    if len(nbits) > 0 and nbits[0] == 1 and all(x == nbits[0] for x in nbits):
        return True
    return False


def generate_inputs(conf):
    """
    Generate sensors inputs form inputs conf :

    a list of (sensor_left, sensor_right)

    :param conf: input conf
    :type conf: dict

    :return: list of sensors pairs
    :rtype: list(tuple(dict, dict))

    """

    # Get needed pairs
    pairs = conf[sens_cst.PAIRING]

    # Generate list of pairs
    list_sensor_pairs = []
    for key1, key2 in pairs:
        merged_key = key1 + "_" + key2
        sensor1 = conf[sens_cst.SENSORS][key1]
        sensor2 = conf[sens_cst.SENSORS][key2]
        list_sensor_pairs.append((merged_key, sensor1, sensor2))

    return list_sensor_pairs
