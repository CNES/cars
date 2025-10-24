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
import math
import os

import numpy as np
import rasterio as rio
from json_checker import Checker, Or

# CARS imports
from cars.core import inputs, projection
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.core.utils import make_relative_path_absolute
from cars.orchestrator.cluster.log_wrapper import cars_profile
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst
from cars.pipelines.parameters.sensor_loaders.sensor_loader import SensorLoader

CARS_GEOID_PATH = "geoid/egm96.grd"  # Path in cars package (pkg)


def sensors_check_inputs(conf, config_dir=None):  # noqa: C901
    """
    Check the inputs given

    :param conf: configuration of inputs
    :type conf: dict
    :param config_dir: path to dir containing json
    :type config_dir: str
    """

    overloaded_conf = conf.copy()

    overloaded_conf[sens_cst.ROI] = conf.get(sens_cst.ROI, None)

    overloaded_conf[sens_cst.PAIRING] = conf.get(sens_cst.PAIRING, None)

    overloaded_conf[sens_cst.INITIAL_ELEVATION] = get_initial_elevation(
        conf.get(sens_cst.INITIAL_ELEVATION, None)
    )

    # Validate inputs
    inputs_schema = {
        sens_cst.SENSORS: dict,
        sens_cst.PAIRING: Or([[str]], None),
        sens_cst.INITIAL_ELEVATION: Or(str, dict, None),
        sens_cst.ROI: Or(str, dict, None),
    }

    checker_inputs = Checker(inputs_schema)
    checker_inputs.validate(overloaded_conf)

    check_sensors(conf, overloaded_conf, config_dir)

    # Check srtm dir
    check_srtm(overloaded_conf[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH])

    return overloaded_conf


def check_sensors(conf, overloaded_conf, config_dir=None):  # noqa: C901
    """
    Check sensors

    """
    # Validate each sensor image
    sensor_schema = {
        sens_cst.INPUT_IMG: Or(str, dict),
        sens_cst.INPUT_GEO_MODEL: Or(str, dict),
        sens_cst.INPUT_MSK: Or(str, None),
        sens_cst.INPUT_CLASSIFICATION: Or(str, dict, None),
    }

    checker_sensor = Checker(sensor_schema)

    for sensor_image_key in conf[sens_cst.SENSORS]:
        # Case where the sensor is defined as a string refering to the input
        # image instead of a dict
        if isinstance(conf[sens_cst.SENSORS][sensor_image_key], str):
            # initialize sensor dictionary
            overloaded_conf[sens_cst.SENSORS][sensor_image_key] = {
                sens_cst.INPUT_IMG: conf[sens_cst.SENSORS][sensor_image_key]
            }

        # Overload parameters
        image = overloaded_conf[sens_cst.SENSORS][sensor_image_key].get(
            sens_cst.INPUT_IMG, None
        )
        if isinstance(image, str):
            loader_name = "basic_image"
        elif isinstance(image, dict):
            loader_name = image.get("loader", "basic_image")
        else:
            raise TypeError(f"Image {image} is not of type str or dict")
        image_loader = SensorLoader(loader_name, image, config_dir)
        image_as_pivot_format = (
            image_loader.get_pivot_format()  # pylint: disable=E1101
        )
        overloaded_conf[sens_cst.SENSORS][sensor_image_key][
            sens_cst.INPUT_IMG
        ] = image_as_pivot_format
        image_path = image_as_pivot_format[sens_cst.MAIN_FILE]

        geomodel = overloaded_conf[sens_cst.SENSORS][sensor_image_key].get(
            "geomodel",
            image_path,
        )
        overloaded_conf[sens_cst.SENSORS][sensor_image_key][
            "geomodel"
        ] = geomodel

        mask = overloaded_conf[sens_cst.SENSORS][sensor_image_key].get(
            sens_cst.INPUT_MSK, None
        )
        overloaded_conf[sens_cst.SENSORS][sensor_image_key][
            sens_cst.INPUT_MSK
        ] = mask

        classif = overloaded_conf[sens_cst.SENSORS][sensor_image_key].get(
            sens_cst.INPUT_CLASSIFICATION, None
        )
        if classif is not None:
            if isinstance(classif, str):
                loader_name = "basic_classif"
            elif isinstance(classif, dict):
                loader_name = classif.get("loader", "basic_classif")
            else:
                raise TypeError(f"Classif {classif} is not of type str or dict")
            classif_loader = SensorLoader(loader_name, classif, config_dir)
            classif_as_pivot_format = (
                classif_loader.get_pivot_format()  # pylint: disable=E1101
            )
            overloaded_conf[sens_cst.SENSORS][sensor_image_key][
                sens_cst.INPUT_CLASSIFICATION
            ] = classif_as_pivot_format
        else:
            overloaded_conf[sens_cst.SENSORS][sensor_image_key][
                sens_cst.INPUT_CLASSIFICATION
            ] = None

        # Validate
        checker_sensor.validate(
            overloaded_conf[sens_cst.SENSORS][sensor_image_key]
        )

    # Modify to absolute path
    if config_dir is not None:
        modify_to_absolute_path(config_dir, overloaded_conf)

    # Check image, msk and color size compatibility
    for sensor_image_key in overloaded_conf[sens_cst.SENSORS]:
        sensor_image = overloaded_conf[sens_cst.SENSORS][sensor_image_key]
        check_input_size(
            sensor_image[sens_cst.INPUT_IMG],
            sensor_image[sens_cst.INPUT_MSK],
            sensor_image[sens_cst.INPUT_CLASSIFICATION],
        )
        # check band nbits of msk
        check_nbits(
            sensor_image[sens_cst.INPUT_MSK],
        )

    # Validate pairs
    # If there is two inputs with no associated pairing, consider that the first
    # image is left and the second image is right
    if (
        overloaded_conf[sens_cst.PAIRING] is None
        and len(overloaded_conf[sens_cst.SENSORS]) == 2
    ):
        sensor_keys = list(overloaded_conf[sens_cst.SENSORS].keys())
        overloaded_conf[sens_cst.PAIRING] = [[sensor_keys[0], sensor_keys[1]]]
        logging.info(
            (
                "Pairing is not defined, '{}' will be used as left sensor and "
                + "'{}' will be used as right sensor"
            ).format(sensor_keys[0], sensor_keys[1])
        )

    if overloaded_conf[sens_cst.PAIRING] is None:
        raise RuntimeError(
            "Pairing is not defined and cannot be determined "
            + "because there are more than two inputs products"
        )

    for key1, key2 in overloaded_conf[sens_cst.PAIRING]:
        if key1 not in overloaded_conf[sens_cst.SENSORS]:
            logging.error("{} not in sensors images".format(key1))
            raise RuntimeError("{} not in sensors images".format(key1))
        if key2 not in overloaded_conf["sensors"]:
            logging.error("{} not in sensors images".format(key2))
            raise RuntimeError("{} not in sensors images".format(key2))

    # Modify to absolute path
    if config_dir is not None:
        modify_to_absolute_path(config_dir, overloaded_conf)
    else:
        logging.debug(
            "path of config file was not given,"
            "relative path are not transformed to absolute paths"
        )

    # Check consistency of pairs images
    for key1, key2 in overloaded_conf[sens_cst.PAIRING]:
        compare_image_type(
            overloaded_conf[sens_cst.SENSORS], sens_cst.INPUT_IMG, key1, key2
        )
        compare_classification_values(
            overloaded_conf[sens_cst.SENSORS],
            sens_cst.INPUT_CLASSIFICATION,
            key1,
            key2,
        )

    return overloaded_conf


def get_sensor_resolution(
    geom_plugin, sensor_path, geomodel, target_epsg=32631
):
    """
    Estimate the sensor image resolution in meters per pixel
    using geolocation of 3 corners of the image.

    :param geom_plugin: geometry plugin instance
    :param sensor_path: path to the sensor image
    :type sensor_path: dict
    :param geomodel: geometric model for the sensor image
    :param target_epsg: target EPSG code for projection
    :type target_epsg: int
    :return: average resolution in meters/pixel along x and y
    :rtype: float
    """
    width, height = inputs.rasterio_get_size(sensor_path[sens_cst.MAIN_FILE])

    upper_left = (0.5, 0.5)
    upper_right = (width - 0.5, 0.5)
    bottom_left = (0.5, height - 0.5)

    # get geodetic coordinates
    lat_ul, lon_ul, _ = geom_plugin.direct_loc(
        sensor_path[sens_cst.MAIN_FILE],
        geomodel,
        np.array([upper_left[0]]),
        np.array([upper_left[1]]),
    )
    lat_ur, lon_ur, _ = geom_plugin.direct_loc(
        sensor_path[sens_cst.MAIN_FILE],
        geomodel,
        np.array([upper_right[0]]),
        np.array([upper_right[1]]),
    )
    lat_bl, lon_bl, _ = geom_plugin.direct_loc(
        sensor_path[sens_cst.MAIN_FILE],
        geomodel,
        np.array([bottom_left[0]]),
        np.array([bottom_left[1]]),
    )

    coords_ll = np.array(
        [[lon_ul, lat_ul, 0], [lon_ur, lat_ur, 0], [lon_bl, lat_bl, 0]]
    )

    # Convert to target CRS
    coords_xy = projection.point_cloud_conversion(coords_ll, 4326, target_epsg)

    diff_x = np.linalg.norm(coords_xy[1] - coords_xy[0])  # UL to UR (width)
    diff_y = np.linalg.norm(coords_xy[2] - coords_xy[0])  # UL to BL (height)

    # resolution in meters per pixel
    res_x = diff_x / (width - 1)
    res_y = diff_y / (height - 1)

    return (res_x + res_y) / 2


def check_geometry_plugin(
    conf_inputs, conf_geom_plugin, epipolar_resolution, output_dem_dir
):
    """
    Check the geometry plugin with inputs

    :param conf_inputs: checked configuration of inputs
    :type conf_inputs: type
    :param conf_advanced: checked configuration of advanced
    :type conf_advanced: type
    :param conf_geom_plugin: name of geometry plugin
    :type conf_geom_plugin: str
    :param epipolar_resolution: epipolar resolution
    :type epipolar_resolution: int
    :return: overload inputs conf
             overloaded geometry plugin conf
             geometry plugin without dem
             geometry plugin with dem
    """
    if conf_geom_plugin is None:
        conf_geom_plugin = "SharelocGeometry"

    # Initialize a temporary plugin, to get the product's resolution
    temp_geom_plugin = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            conf_geom_plugin,
            default_alt=sens_cst.CARS_DEFAULT_ALT,
        )
    )
    average_sensor_resolution = 0
    for _, sensor_image in conf_inputs[sens_cst.SENSORS].items():
        sensor = sensor_image[sens_cst.INPUT_IMG]
        geomodel = sensor_image[sens_cst.INPUT_GEO_MODEL]
        (
            sensor,
            geomodel,
        ) = temp_geom_plugin.check_product_consistency(sensor, geomodel)
        average_sensor_resolution += (
            get_sensor_resolution(temp_geom_plugin, sensor, geomodel)
            * epipolar_resolution
        )
    average_sensor_resolution /= len(conf_inputs[sens_cst.SENSORS])
    # approximate resolution to the highest digit:
    #  0.47 -> 0.5
    #  7.52 -> 8
    #  12.9 -> 10
    nb_digits = int(math.floor(math.log10(abs(average_sensor_resolution))))
    scaling_coeff = round(average_sensor_resolution, -nb_digits)
    # make it so 0.5 (CO3D) is the baseline for parameters
    scaling_coeff *= 2

    # Initialize the desired geometry plugin without elevation information
    geom_plugin_without_dem_and_geoid = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            conf_geom_plugin,
            default_alt=sens_cst.CARS_DEFAULT_ALT,
            scaling_coeff=scaling_coeff,
        )
    )

    # Check products consistency with this plugin
    overloaded_conf_inputs = conf_inputs.copy()
    for sensor_key, sensor_image in conf_inputs[sens_cst.SENSORS].items():
        sensor = sensor_image[sens_cst.INPUT_IMG]
        geomodel = sensor_image[sens_cst.INPUT_GEO_MODEL]
        (
            sensor,
            geomodel,
        ) = geom_plugin_without_dem_and_geoid.check_product_consistency(
            sensor, geomodel
        )
        overloaded_conf_inputs[sens_cst.SENSORS][sensor_key][
            sens_cst.INPUT_IMG
        ] = sensor
        overloaded_conf_inputs[sens_cst.SENSORS][sensor_key][
            sens_cst.INPUT_GEO_MODEL
        ] = geomodel

    geom_plugin_with_dem_and_geoid = generate_geometry_plugin_with_dem(
        conf_geom_plugin,
        conf_inputs,
        scaling_coeff=scaling_coeff,
        output_dem_dir=output_dem_dir,
    )

    return (
        overloaded_conf_inputs,
        conf_geom_plugin,
        geom_plugin_without_dem_and_geoid,
        geom_plugin_with_dem_and_geoid,
        scaling_coeff,
    )


# pylint: disable=too-many-positional-arguments
def generate_geometry_plugin_with_dem(
    conf_geom_plugin,
    conf_inputs,
    dem=None,
    crop_dem=True,
    output_dem_dir=None,
    scaling_coeff=1,
):
    """
    Generate geometry plugin with dem and geoid

    :param conf_geom_plugin: plugin configuration
    :param conf_inputs: inputs configuration
    :param dem: dem to overide the one in inputs
    :param scaling_coeff: scaling factor for resolution
    :type scaling_coeff: float

    :return: geometry plugin object, with a dem
    """

    dem_path = (
        dem
        if dem is not None
        else conf_inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]
    )

    if crop_dem:
        # Get image pairs for DEM intersection with ROI
        pairs_for_roi = []
        for key1, key2 in conf_inputs[sens_cst.PAIRING]:
            sensor1 = conf_inputs[sens_cst.SENSORS][key1]
            sensor2 = conf_inputs[sens_cst.SENSORS][key2]
            image1 = sensor1[sens_cst.INPUT_IMG]
            image2 = sensor2[sens_cst.INPUT_IMG]
            geomodel1 = sensor1[sens_cst.INPUT_GEO_MODEL]
            geomodel2 = sensor2[sens_cst.INPUT_GEO_MODEL]
            pairs_for_roi.append((image1, geomodel1, image2, geomodel2))
    else:
        pairs_for_roi = None

    # Initialize a second geometry plugin with elevation information

    geom_plugin_with_dem_and_geoid = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            conf_geom_plugin,
            dem=dem_path,
            geoid=conf_inputs[sens_cst.INITIAL_ELEVATION][sens_cst.GEOID],
            default_alt=sens_cst.CARS_DEFAULT_ALT,
            pairs_for_roi=pairs_for_roi,
            scaling_coeff=scaling_coeff,
            output_dem_dir=output_dem_dir,
        )
    )

    return geom_plugin_with_dem_and_geoid


def modify_to_absolute_path(config_dir, overloaded_conf):
    """
    Modify input file path to absolute path

    :param config_dir: directory of the json configuration
    :type config_dir: str
    :param overloaded_conf: overloaded configuration json
    :dict overloaded_conf: dict
    """

    for sensor_image_key in overloaded_conf[sens_cst.SENSORS]:
        sensor_image = overloaded_conf[sens_cst.SENSORS][sensor_image_key]
        for tag in [
            sens_cst.INPUT_MSK,
            sens_cst.INPUT_GEO_MODEL,
        ]:
            if isinstance(sensor_image[tag], dict):
                sensor_image[tag][sens_cst.PATH] = make_relative_path_absolute(
                    sensor_image[tag][sens_cst.PATH], config_dir
                )
            elif sensor_image[tag] is not None:
                sensor_image[tag] = make_relative_path_absolute(
                    sensor_image[tag], config_dir
                )

    if overloaded_conf[sens_cst.ROI] is not None:
        if isinstance(overloaded_conf[sens_cst.ROI], str):
            overloaded_conf[sens_cst.ROI] = make_relative_path_absolute(
                overloaded_conf[sens_cst.ROI], config_dir
            )

    for tag in [sens_cst.DEM_PATH, sens_cst.GEOID]:
        if overloaded_conf[sens_cst.INITIAL_ELEVATION][tag] is not None:
            if isinstance(
                overloaded_conf[sens_cst.INITIAL_ELEVATION][tag], str
            ):
                overloaded_conf[sens_cst.INITIAL_ELEVATION][tag] = (
                    make_relative_path_absolute(
                        overloaded_conf[sens_cst.INITIAL_ELEVATION][tag],
                        config_dir,
                    )
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


def get_initial_elevation(config):
    """
    Return initial elevation parameters (dem and geoid paths)
    from input configuration.

    :param config: input initial elevation
    :type config: str, dict or None
    """

    # Case 1 config is already a dict
    if isinstance(config, dict):
        updated_config = config
    else:
        updated_config = {}
        updated_config[sens_cst.DEM_PATH] = (
            config if isinstance(config, str) else None
        )

    # Add geoid path to the initial_elevation dict
    if sens_cst.GEOID not in updated_config:
        # use cars geoid
        logging.info("CARS will use its own internal file as geoid reference")
        # Get root package directory
        package_path = os.path.dirname(__file__)
        geoid_path = os.path.join(
            package_path, "..", "..", "conf", CARS_GEOID_PATH
        )
        updated_config[sens_cst.GEOID] = geoid_path

    return updated_config


def check_input_size(image, mask, classif):
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
    image = image[sens_cst.MAIN_FILE]
    if classif is not None:
        classif = classif[sens_cst.PATH]

    if mask is not None:
        if inputs.rasterio_get_size(image) != inputs.rasterio_get_size(mask):
            raise RuntimeError(
                "The image {} and the mask {} "
                "do not have the same size".format(image, mask)
            )

    if classif is not None:
        if inputs.rasterio_get_size(image) != inputs.rasterio_get_size(classif):
            raise RuntimeError(
                "The classification bands {} and {} "
                "do not have the same size".format(image, classif)
            )


def check_nbits(mask):
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


def compare_image_type(sensors, sensor_type, key1, key2):
    """
    Compare the data type between a pair of images

    :param sensors: list of  sensor paths
    :type sensors: str
    :param sensor_type: type of cardataset image (IMG, MASK, CLASSIF...)
    :type sensor_type: int
    :param key1: key of the images pair
    :type key1: str
    :param key2: other key of the images pair
    :type key2: str
    """
    dtype1 = inputs.rasterio_get_image_type(
        sensors[key1][sensor_type][sens_cst.MAIN_FILE]
    )
    dtype2 = inputs.rasterio_get_image_type(
        sensors[key2][sensor_type][sens_cst.MAIN_FILE]
    )
    if dtype1 != dtype2:
        raise RuntimeError(
            "The pair images haven't the same data type."
            + "\nSensor[{}]: {}".format(key1, dtype1)
            + "; Sensor[{}]: {}".format(key2, dtype2)
        )


def compare_classification_values(sensors, sensor_type, key1, key2):
    """
    Compare the classification values between a pair of images

    :param imgs: list of image paths
    :type imgs: str
    :param classif_type: type of cardataset image (IMG, MASK, CLASSIF...)
    :type classif_type: int
    :param key1: key of the images pair
    :type key1: str
    :param key2: other key of the images pair
    :type key2: str
    """
    classif1 = sensors[key1][sensor_type]
    classif2 = sensors[key2][sensor_type]
    if classif1 is not None and classif2 is not None:
        values1 = classif1[sens_cst.VALUES]
        values2 = classif2[sens_cst.VALUES]
        all_values = list(set(values1) | set(values2))
        classif1[sens_cst.VALUES] = all_values
        classif2[sens_cst.VALUES] = all_values
        filling1 = sensors[key1][sensor_type][sens_cst.FILLING]
        filling2 = sensors[key2][sensor_type][sens_cst.FILLING]
        if filling1 != filling2:
            raise ValueError(
                "Filling rules of {} are not the same as filling "
                "rules of {} but they belong to the same pair".format(
                    classif1[sens_cst.PATH],
                    classif2[sens_cst.PATH],
                )
            )
        filling = filling1
        for filling_method in filling:
            value = filling[filling_method]
            if value is not None and value not in all_values:
                logging.warning(
                    "Value {} on which filling {} must be applied does "
                    "not exist on classifications {} or {}".format(
                        value,
                        filling_method,
                        classif1[sens_cst.PATH],
                        classif2[sens_cst.PATH],
                    )
                )
                logging.warning(
                    "Filling {} is deactivated".format(filling_method)
                )
                filling1[filling_method] = None
                filling2[filling_method] = None


def check_all_nbits_equal_one(nbits):
    """
    Check if all the nbits = 1
    :param nbits: list of the nbits
    :return: True if all the nbits = 1
    """
    if len(nbits) > 0 and nbits[0] == 1 and all(x == nbits[0] for x in nbits):
        return True
    return False


@cars_profile(name="Generate inputs")
def generate_inputs(conf, geometry_plugin):
    """
    Generate sensors inputs form inputs conf :

    a list of (sensor_left, sensor_right)

    :param conf: input conf
    :type conf: dict

    :return: list of sensors pairs
    :rtype: list(tuple(dict, dict))

    """
    # Load geomodels directly on conf object
    sensors = conf[sens_cst.SENSORS]
    for key in sensors:
        geomodel = sensors[key][sens_cst.INPUT_GEO_MODEL]
        loaded_geomodel = geometry_plugin.load_geomodel(geomodel)
        sensors[key][sens_cst.INPUT_GEO_MODEL] = loaded_geomodel

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
