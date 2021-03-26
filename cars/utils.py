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
Utils module:
contains some cars global shared general purpose functions
"""

# Standard imports
from typing import Union, Tuple
import warnings
import os
import logging
import struct
from datetime import datetime
import errno
import numpy as np
import numpy.linalg as la
from json_checker import Checker
import yaml

# Third party imports
import pandas
import xarray as xr
import rasterio as rio
import fiona
from fiona.crs import from_epsg
from shapely.geometry import mapping, shape
import otbApplication

from cars import constants as cst


# Filter rasterio warning when image is not georeferenced
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


def safe_makedirs(directory):
    """
    Create directories even if they already exist (mkdir -p)

    :param directory: path of the directory to create
    """
    try:
        os.makedirs(directory)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(directory):
            pass
        else:
            raise


def make_relative_path_absolute(path, directory):
    """
    If path is a valid relative path with respect to directory,
    returns it as an absolute path

    :param  path: The relative path
    :type path: string
    :param directory: The directory path should be relative to
    :type directory: string
    :returns: os.path.join(directory,path)
        if path is a valid relative path form directory, else path
    :rtype: string
    """
    out = path
    if not os.path.isabs(path):
        abspath = os.path.join(directory, path)
        if os.path.exists(abspath):
            out = abspath
    return out


def ncdf_can_open(file_path):
    """
    Checks if the given file can be opened by NetCDF
    :param file_path: file path.
    :type file_path: str
    :return: True if it can be opened, False otherwise.
    :rtype: bool
    """
    try:
        with xr.open_dataset(file_path) as _:
            return True
    except Exception as read_error:
        logging.error("Exception caught while trying to read file {}: {}"
                      .format(file_path, read_error)
                      )
        return False


def rasterio_can_open(raster_file: str) -> bool:
    """
    Test if a file can be open by rasterio

    :param raster_file: File to test
    :returns: True if rasterio can open file and False otherwise
    """
    try:
        rio.open(raster_file)
        return True
    except Exception as read_error:
        logging.warning("Impossible to read file {}: {}"
                .format(raster_file, read_error))
        return False


def rasterio_get_nb_bands(raster_file: str) -> int:
    """
    Get the number of bands in an image file

    :param f: Image file
    :returns: The number of bands
    """
    with rio.open(raster_file, 'r') as descriptor:
        return descriptor.count


def rasterio_get_size(raster_file: str) -> Tuple[int, int]:
    """
    Get the size of an image (file)

    :param raster_file: Image file
    :returns: The size (width, height)
    """
    with rio.open(raster_file, 'r') as descriptor:
        return (descriptor.width, descriptor.height)

def get_elevation_range_from_metadata(img:str, default_min:float=0,
                                default_max:float=300) -> Tuple[float, float]:
    """
    This function will try to derive a valid RPC altitude range
    from img metadata.

    It will first try to read metadata with gdal.
    If it fails, it will look for values in the geom file if it exists
    If it fails, it will return the default range

    :param img: Path to the img for which the elevation range is required
    :param default_min: Default minimum value to return if everything else fails
    :param default_max: Default minimum value to return if everything else fails
    :returns: (elev_min, elev_max) float tuple
    """
    # First, try to get range from gdal metadata
    with rio.open(img) as descriptor:
        gdal_height_offset = descriptor.get_tag_item('HEIGHT_OFF','RPC')
        gdal_height_scale  = descriptor.get_tag_item('HEIGHT_SCALE','RPC')

        if gdal_height_scale is not None and gdal_height_offset is not None:
            if isinstance(gdal_height_offset, str):
                gdal_height_offset = float(gdal_height_offset)
            if isinstance(gdal_height_scale, str):
                gdal_height_scale = float(gdal_height_scale)
            return (float(gdal_height_offset-gdal_height_scale/2.),
                    float(gdal_height_offset+gdal_height_scale/2.))

    # If we are still here, try to get range from OTB/OSSIM geom file if exists
    geom_file, _ = os.path.splitext(img)
    geom_file = geom_file+".geom"

    # If geom file exists
    if os.path.isfile(geom_file):
        with open(geom_file,'r') as geom_file_desc:
            geom_height_offset = None
            geom_height_scale = None

            for line in geom_file_desc:
                if line.startswith("height_off:"):
                    geom_height_offset = float(line.split(sep=':')[1])

                if line.startswith("height_scale:"):
                    geom_height_scale = float(line.split(sep=':')[1])
            if geom_height_offset is not None and geom_height_scale is not None:
                return (float(geom_height_offset-geom_height_scale/2),
                        float(geom_height_offset+geom_height_scale/2))

    # If we are still here, return a default range:
    return (default_min, default_max)

def otb_can_open(raster_file: str) -> bool:
    """
    Test if file can be open by otb
    and that it has a correct geom file associated

    :param raster_file: filename
    :return: True if the file can be used with the otb, False otherwise
    """
    read_im_app = otbApplication.Registry.CreateApplication("ReadImageInfo")
    read_im_app.SetParameterString("in", raster_file)
    read_im_app.SetParameterString("outkwl", "./otb_can_open_test.geom")

    try:
        read_im_app.ExecuteAndWriteOutput()
        if os.path.exists("./otb_can_open_test.geom"):
            with open("./otb_can_open_test.geom") as geom_file_desc:
                geom_dict = dict()
                for line in geom_file_desc:
                    key, val = line.split(': ')
                    geom_dict[key] = val
                #pylint: disable=too-many-boolean-expressions
                if      'line_den_coeff_00' not in geom_dict or \
                        'samp_den_coeff_00' not in geom_dict or \
                        'line_num_coeff_00' not in geom_dict or \
                        'samp_num_coeff_00' not in geom_dict or \
                        'line_off' not in geom_dict or \
                        'line_scale' not in geom_dict or \
                        'samp_off' not in geom_dict or \
                        'samp_scale' not in geom_dict or \
                        'lat_off' not in geom_dict or \
                        'lat_scale' not in geom_dict or \
                        'long_off' not in geom_dict or \
                        'long_scale' not in geom_dict or \
                        'height_off' not in geom_dict or \
                        'height_scale' not in geom_dict or \
                        'polynomial_format' not in geom_dict:
                    logging.error("No RPC model set for image {}"
                                .format(geom_file_desc))
                    return False

            os.remove("./otb_can_open_test.geom")
            return True
        # else
        logging.error("{} does not have associated geom file"
                        .format(geom_file_desc))
        return False
    except Exception as read_error:
        logging.error(
            "Exception caught while trying to read file {}: {}"
                .format(raster_file, read_error))
        return False

def read_vector(path_to_file):
    """
    Read vector file and returns the corresponding polygon

    :raise Exception when the input file is unreadable

    :param path_to_file: path to the file to open
    :type path_to_file: str
    :return: a shapely polygon
    :rtype: tuple (polygon, epsg)
    """
    try:
        polys = []
        with fiona.open(path_to_file) as vec_file:
            _, epsg = vec_file.crs['init'].split(':')
            for feat in vec_file:
                polys.append(shape(feat['geometry']))
    except BaseException as base_except:
        raise Exception('Impossible to read {} file'.format(
                        path_to_file)) from base_except

    if len(polys) == 1:
        return polys[0], int(epsg)

    if len(polys) > 1:
        logging.info('Multi features files are not supported, '
                     'the first feature of {} will be used'.
                     format(path_to_file))
        return polys[0], int(epsg)

    logging.info(
        'No feature is present in the {} file'.format(path_to_file))
    return None


def write_vector(polys, path_to_file, epsg, driver='GPKG'):
    """
    Write list of polygons in a single vector file

    :param polys: list of polygons to write in the file
    :param path_to_file: file to create
    :param epsg: EPSG code of the polygons
    :param driver: vector file type (default format is geopackage)
    """
    crs = from_epsg(epsg)
    sch = {
        'geometry': 'Polygon',
        'properties': {
            'Type': 'str:10'
        }
    }

    with fiona.open(path_to_file, 'w',
                    crs=crs, driver=driver, schema=sch) as vector_file:
        for poly in polys:
            poly_dict = {
                'geometry': mapping(poly),
                'properties': {
                    'Type': 'Polygon'
                }
            }
            vector_file.write(poly_dict)

def angle_vectors(vector_1: np.ndarray, vector_2: np.ndarray) -> float :
    """
    Compute the smallest angle in radians between two angle_vectors
    Use arctan2 more precise than arcos2
    Tan θ = |(axb)|/ (a.b)
    (same: Cos θ = (a.b)/(|a||b|))

    :param vector_1: Numpy first vector
    :param vector_2: Numpy second vector
    :return: Smallest angle in radians
    """

    vec_dot = np.dot(vector_1, vector_2)
    vec_norm = la.norm(np.cross(vector_1, vector_2))
    return np.arctan2(vec_norm, vec_dot)


def write_ply(path_ply_file: str, cloud: Union[xr.Dataset, pandas.DataFrame]):
    """
    Write cloud to a ply file

    :param path: path to the ply file to write
    :param cloud: cloud to write,
        it can be a xr.Dataset as the ones given in output of the triangulation
    or a pandas.DataFrame as used in the rasterization
    """

    with open(path_ply_file, 'w') as ply_file:
        if isinstance(cloud, xr.Dataset):
            nb_points = int(cloud[cst.POINTS_CLOUD_CORR_MSK]
                .where(cloud[cst.POINTS_CLOUD_CORR_MSK].values != 0).count())
        else:
            nb_points = cloud.shape[0]

        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex {}\n".format(nb_points))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")

        if isinstance(cloud, xr.Dataset):
            for x_item, y_item, z_item, mask_item in zip(
                            np.nditer(cloud[cst.X].values),
                            np.nditer(cloud[cst.Y].values),
                            np.nditer(cloud[cst.Z].values),
                            np.nditer(cloud[cst.POINTS_CLOUD_CORR_MSK].values)):
                if mask_item != 0:
                    ply_file.write("{} {} {}\n".format(x_item, y_item, z_item))
        else:
            for xyz in cloud.itertuples():
                ply_file.write("{} {} {}\n".format(getattr(xyz, cst.X),
                                            getattr(xyz, cst.Y),
                                            getattr(xyz, cst.Z)))


def read_geoid_file():
    """
    Read geoid height from OTB geoid file
    Geoid is defined by the $OTB_GEOID_FILE global environement variable.

    A default CARS geoid is deployed in setup.py and
    configured in configuration.py

    Geoid is returned as an xarray.Dataset and height is stored in the `hgt`
    variable, which is indexed by `lat` and `lon` coordinates. Dataset
    attributes contain geoid bounds geodetic coordinates and
    latitude/longitude step spacing.

    :return: the geoid height array in meter.
    :rtype: xarray.Dataset
    """
    # Set geoid path from OTB_GEOID_FILE
    geoid_path = os.environ.get('OTB_GEOID_FILE')

    with open(geoid_path, mode='rb') as in_grd:  # reading binary data
        # first header part, 4 float of 4 bytes -> 16 bytes to read
        # Endianness seems to be Big-Endian.
        lat_min, lat_max, lon_min, lon_max = struct.unpack('>ffff',
                                                           in_grd.read(16))
        lat_step, lon_step = struct.unpack('>ff', in_grd.read(8))

        n_lats = int(np.ceil((lat_max - lat_min)) / lat_step) + 1
        n_lons = int(np.ceil((lon_max - lon_min)) / lon_step) + 1

        # read height grid.
        geoid_height = np.fromfile(in_grd, '>f4').reshape(n_lats, n_lons)

        # create output Dataset
        geoid = xr.Dataset({'hgt': (('lat', 'lon'), geoid_height)},
                           coords={
                               'lat': np.linspace(lat_max, lat_min, n_lats),
                               'lon': np.linspace(lon_min, lon_max, n_lons)},
                           attrs={'lat_min': lat_min, 'lat_max': lat_max,
                                  'lon_min': lon_min, 'lon_max': lon_max,
                                  'd_lat': lat_step, 'd_lon': lon_step}
                           )

        return geoid

def add_log_file(out_dir, command):
    """
    Add dated file handler to the logger.

    :param out_dir: output directory in which the log file will be created
    :type out_dir: str
    :param command: command name which will be part of the log file name
    :type command: str
    """
    # set file log handler
    now = datetime.now()
    h_log_file = logging.FileHandler(os.path.join(out_dir,
                '{}_{}.log'.format(now.strftime("%y-%m-%d_%Hh%Mm"), command)))
    h_log_file.setLevel(logging.getLogger().getEffectiveLevel())

    formatter = logging.Formatter(
        fmt='%(asctime)s :: %(levelname)s :: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )
    h_log_file.setFormatter(formatter)

    # add it to the logger
    logging.getLogger().addHandler(h_log_file)


def check_json(conf, schema):
    """
    Check a dictionary with respect to a schema

    :param conf: The dictionary to check
    :type conf: dict
    :param schema: The schema to use
    :type schema: dict

    :returns: conf if check succeeds (else raises CheckerError)
    :rtype: dict
    """
    schema_validator = Checker(schema)
    checked_conf = schema_validator.validate(conf)
    return checked_conf


def write_dask_config(dask_config: dict,
        output_dir: str,
        file_name: str):
    """
    Writes the dask config used in yaml format.

    :param dask_config: Dask config used
    :type dask_config: dict
    :param output_dir: output directory path
    :type dask_config: dict
    :param output_dir: output directory path
    """

    # warning
    logging.info("Dask will merge several config files"\
        "located at default locations such as"\
        " ~/.config/dask/ .\n Dask config in "\
        " $DASK_DIR will be used with the highest priority.")

    # file path where to store the dask config
    dask_config_path = os.path.join(output_dir, file_name + ".yaml")
    with open(dask_config_path, 'w') as dask_config_file:
        yaml.dump(dask_config, dask_config_file)
