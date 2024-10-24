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
from cars.core import inputs, preprocessing, roi_tools
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.core.utils import make_relative_path_absolute
from cars.pipelines.parameters import advanced_parameters_constants as adv_cst
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst

CARS_GEOID_PATH = "geoid/egm96.grd"  # Path in cars package (pkg)


def sensors_check_inputs(conf, config_json_dir=None):  # noqa: C901
    """
    Check the inputs given

    :param conf: configuration of inputs
    :type conf: dict
    :param config_json_dir: path to dir containing json
    :type config_json_dir: str
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

    # Validate each sensor image
    sensor_schema = {
        sens_cst.INPUT_IMG: str,
        sens_cst.INPUT_COLOR: str,
        sens_cst.INPUT_NODATA: int,
        sens_cst.INPUT_GEO_MODEL: Or(str, dict),
        sens_cst.INPUT_MSK: Or(str, None),
        sens_cst.INPUT_CLASSIFICATION: Or(str, None),
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

        # Overload optional parameters
        geomodel = overloaded_conf[sens_cst.SENSORS][sensor_image_key].get(
            "geomodel",
            overloaded_conf[sens_cst.SENSORS][sensor_image_key][
                sens_cst.INPUT_IMG
            ],
        )
        overloaded_conf[sens_cst.SENSORS][sensor_image_key][
            "geomodel"
        ] = geomodel

        color = overloaded_conf[sens_cst.SENSORS][sensor_image_key].get(
            "color",
            overloaded_conf[sens_cst.SENSORS][sensor_image_key][
                sens_cst.INPUT_IMG
            ],
        )
        overloaded_conf[sens_cst.SENSORS][sensor_image_key][
            sens_cst.INPUT_COLOR
        ] = color

        no_data = overloaded_conf[sens_cst.SENSORS][sensor_image_key].get(
            sens_cst.INPUT_NODATA, 0
        )
        overloaded_conf[sens_cst.SENSORS][sensor_image_key][
            sens_cst.INPUT_NODATA
        ] = no_data

        mask = overloaded_conf[sens_cst.SENSORS][sensor_image_key].get(
            sens_cst.INPUT_MSK, None
        )
        overloaded_conf[sens_cst.SENSORS][sensor_image_key][
            sens_cst.INPUT_MSK
        ] = mask
        classif = overloaded_conf[sens_cst.SENSORS][sensor_image_key].get(
            sens_cst.INPUT_CLASSIFICATION, None
        )
        overloaded_conf[sens_cst.SENSORS][sensor_image_key][
            sens_cst.INPUT_CLASSIFICATION
        ] = classif
        # Validate
        checker_sensor.validate(
            overloaded_conf[sens_cst.SENSORS][sensor_image_key]
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
    if config_json_dir is not None:
        modify_to_absolute_path(config_json_dir, overloaded_conf)

    else:
        logging.debug(
            "path of config file was not given,"
            "relative path are not transformed to absolute paths"
        )

    # check datat type of pairs images
    for key1, key2 in overloaded_conf[sens_cst.PAIRING]:
        compare_image_type(
            overloaded_conf[sens_cst.SENSORS], sens_cst.INPUT_IMG, key1, key2
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
        check_input_data(
            sensor_image[sens_cst.INPUT_IMG],
            sensor_image[sens_cst.INPUT_COLOR],
        )

    # Check srtm dir
    check_srtm(overloaded_conf[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH])

    return overloaded_conf


def check_geometry_plugin(conf_inputs, conf_advanced, conf_geom_plugin):
    """
    Check the geometry plugin with inputs

    :param conf_inputs: checked configuration of inputs
    :type conf_inputs: type
    :param cond_advanced: checked configuration of advanced
    :type conf_advanced: type
    :param conf_geom_plugin: name of geometry plugin
    :type conf_geom_plugin: str
    :return: overload inputs conf
             overloaded geometry plugin conf
             geometry plugin without dem
             geometry plugin with dem
    """
    if conf_geom_plugin is None:
        conf_geom_plugin = "SharelocGeometry"

    # Initialize the desired geometry plugin without elevation information
    geom_plugin_without_dem_and_geoid = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            conf_geom_plugin, default_alt=sens_cst.CARS_DEFAULT_ALT
        )
    )

    # If use a priori, override initial elevation with dem_median
    if adv_cst.USE_EPIPOLAR_A_PRIORI in conf_advanced:
        if conf_advanced[adv_cst.USE_EPIPOLAR_A_PRIORI]:
            if adv_cst.DEM_MEDIAN in conf_advanced[adv_cst.TERRAIN_A_PRIORI]:
                conf_inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH] = (
                    conf_advanced[adv_cst.TERRAIN_A_PRIORI][adv_cst.DEM_MEDIAN]
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
        conf_geom_plugin, conf_inputs
    )

    # Check dem is big enough
    dem_generation_roi_poly = None
    needed_dem_roi = geom_plugin_with_dem_and_geoid.dem_roi
    needed_dem_roi_epsg = geom_plugin_with_dem_and_geoid.dem_roi_epsg
    if needed_dem_roi is not None:
        needed_dem_roi_poly = roi_tools.bounds_to_poly(needed_dem_roi)
        # convert to 4326 roi
        dem_generation_roi_poly = preprocessing.compute_roi_poly(
            needed_dem_roi_poly, needed_dem_roi_epsg, 4326
        )

        if (
            conf_inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]
            is not None
        ):
            # get dem total roi
            total_input_roi_poly = roi_tools.bounds_to_poly(
                inputs.rasterio_get_bounds(
                    conf_inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]
                )
            )
            total_input_roi_epsg = inputs.rasterio_get_epsg_code(
                conf_inputs[sens_cst.INITIAL_ELEVATION][sens_cst.DEM_PATH]
            )
            total_input_roi_poly = preprocessing.compute_roi_poly(
                total_input_roi_poly, total_input_roi_epsg, 4326
            )

            # if needed roi not inside dem, raise error
            if not total_input_roi_poly.contains_properly(
                dem_generation_roi_poly
            ):
                raise RuntimeError(
                    "Given initial elevation ROI is not covering needed ROI: "
                    " EPSG:4326, ROI: {}".format(dem_generation_roi_poly.bounds)
                )

    else:
        logging.warning(
            "Current geometry plugin doesnt compute dem roi needed "
            "for later computations. Errors related to unsufficient "
            "dem roi might occur."
        )

    return (
        overloaded_conf_inputs,
        conf_geom_plugin,
        geom_plugin_without_dem_and_geoid,
        geom_plugin_with_dem_and_geoid,
        dem_generation_roi_poly,
    )


def generate_geometry_plugin_with_dem(
    conf_geom_plugin, conf_inputs, dem=None, crop_dem=True
):
    """
    Generate geometry plugin with dem and geoid

    :param conf_geom_plugin: plugin configuration
    :param conf_inputs: inputs configuration
    :param dem: dem to overide the one in inputs

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
        )
    )

    return geom_plugin_with_dem_and_geoid


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
            if isinstance(sensor_image[tag], dict):
                sensor_image[tag]["path"] = make_relative_path_absolute(
                    sensor_image[tag]["path"], config_json_dir
                )
            elif sensor_image[tag] is not None:
                sensor_image[tag] = make_relative_path_absolute(
                    sensor_image[tag], config_json_dir
                )

    if overloaded_conf[sens_cst.ROI] is not None:
        if isinstance(overloaded_conf[sens_cst.ROI], str):
            overloaded_conf[sens_cst.ROI] = make_relative_path_absolute(
                overloaded_conf[sens_cst.ROI], config_json_dir
            )

    for tag in [sens_cst.DEM_PATH, sens_cst.GEOID]:
        if overloaded_conf[sens_cst.INITIAL_ELEVATION][tag] is not None:
            if isinstance(
                overloaded_conf[sens_cst.INITIAL_ELEVATION][tag], str
            ):
                overloaded_conf[sens_cst.INITIAL_ELEVATION][tag] = (
                    make_relative_path_absolute(
                        overloaded_conf[sens_cst.INITIAL_ELEVATION][tag],
                        config_json_dir,
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


def compare_image_type(imgs, image_type, key1, key2):
    """
    Compare the data type between a pair of images

    :param imgs: list of image paths
    :type imgs: str
    :param key1: key of the images pair
    :type key1: str
    :param image_type: type of cardataset image (IMG, MASK, CLASSIF...)
    :type image_type: int
    :param key1: other key of the images pair
    :type key1: str
    """
    dtype1 = inputs.rasterio_get_image_type(imgs[key1][image_type])
    dtype2 = inputs.rasterio_get_image_type(imgs[key2][image_type])
    if dtype1 != dtype2:
        raise RuntimeError(
            "The pair images haven't the same data type."
            + "\nSensor[{}]: {}".format(key1, dtype1)
            + "; Sensor[{}]: {}".format(key2, dtype2)
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
