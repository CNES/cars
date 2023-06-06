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
Projection module:
contains some general purpose functions using polygons and data projections
"""
# pylint: disable=too-many-lines

# Standard imports
import logging
import math
import os
from typing import List, Tuple, Union

# Third party imports
import numpy as np
import pandas
import pyproj
import rasterio as rio
import xarray as xr
from rasterio.features import shapes
from shapely.geometry import Polygon, shape
from shapely.ops import transform

from cars.conf import input_parameters
from cars.core import constants as cst
from cars.core import inputs, outputs, utils

# CARS imports
from cars.core.geometry import AbstractGeometry


def compute_dem_intersection_with_poly(
    srtm_dir: str, ref_poly: Polygon, ref_epsg: int
) -> Polygon:
    """
    Compute the intersection polygon between the defined dem regions
    and the reference polygon in input

    :raise Exception: when the input dem doesn't intersect the reference polygon

    :param srtm_dir: srtm directory
    :param ref_poly: reference polygon
    :param ref_epsg: reference epsg code
    :return: The intersection polygon between the defined dem regions
        and the reference polygon in input
    """
    dem_poly = None
    for _, _, srtm_files in os.walk(srtm_dir):
        logging.info(
            "Browsing all files of the srtm dir. "
            "Some files might be unreadable by rasterio (non blocking matter)."
        )
        for file in srtm_files:
            unsupported_formats = [".omd"]
            _, ext = os.path.splitext(file)
            if ext not in unsupported_formats:
                if inputs.rasterio_can_open(os.path.join(srtm_dir, file)):
                    with rio.open(os.path.join(srtm_dir, file)) as data:
                        xmin = min(data.bounds.left, data.bounds.right)
                        ymin = min(data.bounds.bottom, data.bounds.top)
                        xmax = max(data.bounds.left, data.bounds.right)
                        ymax = max(data.bounds.bottom, data.bounds.top)

                        try:
                            file_epsg = data.crs.to_epsg()
                            file_bb = Polygon(
                                [
                                    (xmin, ymin),
                                    (xmin, ymax),
                                    (xmax, ymax),
                                    (xmax, ymin),
                                    (xmin, ymin),
                                ]
                            )

                            # transform polygon if needed
                            if ref_epsg != file_epsg:
                                file_bb = polygon_projection(
                                    file_bb, file_epsg, ref_epsg
                                )

                            # if the srtm tile intersects the reference polygon
                            if file_bb.intersects(ref_poly):
                                local_dem_poly = None

                                # retrieve valid polygons
                                for poly, val in shapes(
                                    data.dataset_mask(),
                                    transform=data.transform,
                                ):
                                    if val != 0:
                                        poly = shape(poly)
                                        poly = poly.buffer(0)
                                        if ref_epsg != file_epsg:
                                            poly = polygon_projection(
                                                poly, file_epsg, ref_epsg
                                            )

                                        # combine valid polygons
                                        if local_dem_poly is None:
                                            local_dem_poly = poly
                                        else:
                                            local_dem_poly = poly.union(
                                                local_dem_poly
                                            )

                                # combine the tile valid polygon to the other
                                # tiles' ones
                                if dem_poly is None:
                                    dem_poly = local_dem_poly
                                else:
                                    dem_poly = dem_poly.union(local_dem_poly)

                        except AttributeError as attribute_error:
                            logging.warning(
                                "Impossible to read the SRTM"
                                "tile epsg code: {}".format(attribute_error)
                            )

    # compute dem coverage polygon over the reference polygon
    if dem_poly is None or not dem_poly.intersects(ref_poly):
        raise RuntimeError("The input DEM does not intersect the useful zone")

    dem_cover = dem_poly.intersection(ref_poly)

    area_cover = dem_cover.area
    area_inter = ref_poly.area

    return dem_cover, area_cover / area_inter * 100.0


def polygon_projection(poly: Polygon, epsg_in: int, epsg_out: int) -> Polygon:
    """
    Projects a polygon from an initial epsg code to another

    :param poly: poly to project
    :param epsg_in: initial epsg code
    :param epsg_out: final epsg code
    :return: The polygon in the final projection
    """
    # Get CRS from input EPSG codes
    crs_in = pyproj.CRS.from_epsg(epsg_in)
    crs_out = pyproj.CRS.from_epsg(epsg_out)
    # Project polygon between CRS (keep always_xy for compatibility)
    project = pyproj.Transformer.from_crs(crs_in, crs_out, always_xy=True)
    poly = transform(project.transform, poly)

    return poly


def geo_to_ecef(
    lat: np.ndarray, lon: np.ndarray, alt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Point transformation from Geodetic of ellipsoid WGS-84) to ECEF
    ECEF: Earth-centered, Earth-fixed

    :param lat: input geodetic latitude (angle in degree)
    :param lon:  input geodetic longitude (angle in degree)
    :param alt: input altitude above geodetic ellipsoid (meters)
    :return:  ECEF (Earth centered, Earth fixed) x, y, z coordinates tuple
                                                    (in meters)
    """
    epsg_in = 4979  # EPSG code for Geocentric WGS84 in lat, lon, alt (degree)
    epsg_out = 4978  # EPSG code for ECEF WGS84 in x, y, z (meters)

    return points_cloud_conversion(
        np.array([[lon, lat, alt]]), epsg_in, epsg_out
    )[0]


def ecef_to_enu(
    x_ecef: np.ndarray,
    y_ecef: np.ndarray,
    z_ecef: np.ndarray,
    lat0: np.ndarray,
    lon0: np.ndarray,
    alt0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Coordinates conversion from ECEF Earth Centered to
    East North Up Coordinate from a reference point (lat0, lon0, alt0)

    See Wikipedia page for details:
    https://en.wikipedia.org/wiki/Geographic_coordinate_conversion

    :param x_ecef: target x ECEF coordinate (meters)
    :param y_ecef: target y ECEF coordinate (meters)
    :param z_ecef: target z ECEF coordinate (meters)
    :param lat0: Reference geodetic latitude
    :param lon0: Reference geodetic longitude
    :param alt0: Reference altitude above geodetic ellipsoid (meters)
    :return: ENU (xEast, yNorth zUp) target coordinates tuple (meters)
    """
    # Intermediate computing for ENU conversion
    cos_lat0 = np.cos(np.radians(lat0))
    sin_lat0 = np.sin(np.radians(lat0))

    cos_long0 = np.cos(np.radians(lon0))
    sin_long0 = np.sin(np.radians(lon0))

    # Determine ECEF coordinates from reference geodetic
    x0_ecef, y0_ecef, z0_ecef = geo_to_ecef(lat0, lon0, alt0)

    x_east = (-(x_ecef - x0_ecef) * sin_long0) + (
        (y_ecef - y0_ecef) * cos_long0
    )
    y_north = (
        (-cos_long0 * sin_lat0 * (x_ecef - x0_ecef))
        - (sin_lat0 * sin_long0 * (y_ecef - y0_ecef))
        + (cos_lat0 * (z_ecef - z0_ecef))
    )
    z_up = (
        (cos_lat0 * cos_long0 * (x_ecef - x0_ecef))
        + (cos_lat0 * sin_long0 * (y_ecef - y0_ecef))
        + (sin_lat0 * (z_ecef - z0_ecef))
    )

    return x_east, y_north, z_up


def geo_to_enu(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    lat0: np.ndarray,
    lon0: np.ndarray,
    alt0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Point transformation from WGS-84 Geodetic coordinates to to ENU.
    Use geo_to_ecef and ecef_to_enu functions.

    :param lat: input geodetic latitude (angle in degree)
    :param lon:  input geodetic longitude (angle in degree)
    :param alt: input altitude above geodetic ellipsoid (meters)
    :param lat0: Reference geodetic latitude
    :param lon0: Reference geodetic longitude
    :param alt0: Reference altitude above geodetic ellipsoid (meters)
    :return: ENU (xEast, yNorth zUp) target coordinates tuple (meters)
    """
    x_ecef, y_ecef, z_ecef = geo_to_ecef(lat, lon, alt)
    return ecef_to_enu(x_ecef, y_ecef, z_ecef, lat0, lon0, alt0)


def enu_to_aer(
    x_east: np.ndarray, y_north: np.ndarray, z_up: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ENU coordinates to Azimuth, Elevation angle, Range from ENU origin
    Beware: Elevation angle is not the altitude.

    :param x_east: ENU East coordinate (meters)
    :param y_north: ENU North coordinate (meters)
    :param z_up: ENU Up coordinate (meters)
    :return: Azimuth, Elevation Angle, Slant Range (degrees, degrees, meters)
    """

    xy_range = np.hypot(x_east, y_north)  # Distance of e, n vector
    xyz_range = np.hypot(xy_range, z_up)  # Distance of e, n, u vector
    elevation = np.arctan2(z_up, xy_range)
    azimuth = np.arctan2(x_east, y_north) % (2 * np.pi)
    # From [-pi,+pi] to [0,2pi]

    azimuth = np.degrees(azimuth)
    elevation = np.degrees(elevation)

    return azimuth, elevation, xyz_range


def geo_to_aer(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    lat0: np.ndarray,
    lon0: np.ndarray,
    alt0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gives Azimuth, Elevation angle and Slant Range
    from a Reference to a Point with geodetic coordinates.

    :param lat: input geodetic latitude (angle in degree)
    :param lon: input geodetic longitude (angle in degree)
    :param alt: input altitude above geodetic ellipsoid (meters)
    :param lat0: Reference geodetic latitude
    :param lon0: Reference geodetic longitude
    :param alt0: Reference altitude above geodetic ellipsoid (meters)
    :return: Azimuth, Elevation Angle, Slant Range (degrees, degrees, meters)
    """
    x_east, y_north, z_up = geo_to_enu(lat, lon, alt, lat0, lon0, alt0)
    return enu_to_aer(x_east, y_north, z_up)


def points_cloud_conversion(
    cloud_in: np.ndarray, epsg_in: int, epsg_out: int
) -> np.ndarray:
    """
    Convert a point cloud from a SRS to another one.

    :param cloud_in: cloud to project
    :param epsg_in: EPSG code of the input SRS
    :param epsg_out: EPSG code of the output SRS
    :return: Projected point cloud
    """
    # Get CRS from input EPSG codes
    crs_in = pyproj.CRS.from_epsg(epsg_in)
    crs_out = pyproj.CRS.from_epsg(epsg_out)

    # Project point cloud between CRS (keep always_xy for compatibility)
    cloud_in = np.array(cloud_in).T
    transformer = pyproj.Transformer.from_crs(crs_in, crs_out, always_xy=True)
    cloud_in = transformer.transform(*cloud_in)
    cloud_in = np.array(cloud_in).T

    return cloud_in


def get_xyz_np_array_from_dataset(
    cloud_in: xr.Dataset,
) -> Tuple[np.array, List[int]]:
    """
    Get a numpy array of size (nb_points, 3) with the columns
    being the x, y and z coordinates from a dataset as given
    in output of the triangulation.

    The original epipolar geometry shape is also given in output
    in order to reshape the output numpy array in its
    original geometry if necessary.

    :param cloud_in: input xarray dataset
    :return: a tuple composed of the xyz numàpy array and its original shape
    """
    xyz = np.stack(
        (cloud_in[cst.X].values, cloud_in[cst.Y].values, cloud_in[cst.Z]),
        axis=-1,
    )
    xyz_shape = xyz.shape
    xyz = np.reshape(xyz, (-1, 3))

    return xyz, xyz_shape


def get_converted_xy_np_arrays_from_dataset(
    cloud_in: xr.Dataset, epsg_out: int
) -> Tuple[np.array, np.array]:
    """
    Get the x and y coordinates as numpy array
    in the new referential indicated by epsg_out.
    TODO: add test

    :param cloud_in: input xarray dataset
    :param epsg_out: target epsg code
    :return: a tuple composed of the x and y numpy arrays
    """
    xyz, xyz_shape = get_xyz_np_array_from_dataset(cloud_in)
    epsg = int(cloud_in.attrs[cst.EPSG])
    xyz = points_cloud_conversion(xyz, epsg, epsg_out)
    xyz = xyz.reshape(xyz_shape)
    if isinstance(cloud_in, xr.Dataset):
        proj_x = xyz[:, :, 0]
        proj_y = xyz[:, :, 1]
    else:
        proj_x = xyz[:, 0]
        proj_y = xyz[:, 1]
    return proj_x, proj_y


def points_cloud_conversion_dataset(cloud: xr.Dataset, epsg_out: int):
    """
    Convert a point cloud as an xarray.Dataset to another epsg (inplace)
    TODO: add test

    :param cloud: cloud to project
    :param epsg_out: EPSG code of the output SRS
    """

    if cloud.attrs[cst.EPSG] != epsg_out:
        xyz, xyz_shape = get_xyz_np_array_from_dataset(cloud)

        xyz = points_cloud_conversion(xyz, cloud.attrs[cst.EPSG], epsg_out)
        xyz = xyz.reshape(xyz_shape)
        if isinstance(cloud, xr.Dataset):
            # # Update cloud_in x, y and z values
            cloud[cst.X].values = xyz[:, :, 0]
            cloud[cst.Y].values = xyz[:, :, 1]
            cloud[cst.Z].values = xyz[:, :, 2]

            # # Update EPSG code
            cloud.attrs[cst.EPSG] = epsg_out
        elif isinstance(cloud, pandas.DataFrame):
            cloud[cst.X] = xyz[:, 0]
            cloud[cst.Y] = xyz[:, 1]
            cloud[cst.Z] = xyz[:, 2]
            cloud.attrs[cst.EPSG] = epsg_out
        else:
            logging.error(
                "points_cloud_conversion_dataset error: point cloud is unknown"
            )


def points_cloud_conversion_dataframe(
    cloud: pandas.DataFrame, epsg_in: int, epsg_out: int
):
    """
    Convert a point cloud as a panda.DataFrame to another epsg (inplace)

    :param cloud: cloud to project
    :param epsg_in: EPSG code of the input SRS
    :param epsg_out: EPSG code of the output SRS
    """
    xyz_in = cloud.loc[:, [cst.X, cst.Y, cst.Z]].values

    if xyz_in.shape[0] != 0:
        xyz_in = points_cloud_conversion(xyz_in, epsg_in, epsg_out)
        cloud[cst.X] = xyz_in[:, 0]
        cloud[cst.Y] = xyz_in[:, 1]
        cloud[cst.Z] = xyz_in[:, 2]


def ground_polygon_from_envelopes(
    poly_envelope1: Polygon,
    poly_envelope2: Polygon,
    epsg1: int,
    epsg2: int,
    tgt_epsg: int = 4326,
) -> Tuple[Polygon, Tuple[int, int, int, int]]:
    """
    compute the ground polygon of the intersection of two envelopes
    TODO: refacto with externals (OTB) and steps.

    :raise: Exception when the envelopes don't intersect one to each other

    :param poly_envelope1: path to the first envelope
    :param poly_envelope2: path to the second envelope
    :param epsg1: EPSG code of poly_envelope1
    :param epsg2: EPSG code of poly_envelope2
    :param tgt_epsg: EPSG code of the new projection
           (default value is set to 4326)
    :return: a tuple with the shapely polygon of the intersection
             and the intersection's bounding box
             (described by a tuple (minx, miny, maxx, maxy))
    """
    # project to the correct epsg if necessary
    if epsg1 != tgt_epsg:
        poly_envelope1 = polygon_projection(poly_envelope1, epsg1, tgt_epsg)

    if epsg2 != tgt_epsg:
        poly_envelope2 = polygon_projection(poly_envelope2, epsg2, tgt_epsg)

    # intersect both envelopes
    if poly_envelope1.intersects(poly_envelope2):
        inter = poly_envelope1.intersection(poly_envelope2)
    else:
        raise RuntimeError("The two envelopes do not intersect one another")

    return inter, inter.bounds


def ground_intersection_envelopes(
    conf,
    geometry_loader_to_use: str,
    shp1_path: str,
    shp2_path: str,
    out_intersect_path: str,
    geoid: str = None,
    dem_dir: str = None,
    default_alt: float = None,
) -> Tuple[Polygon, Tuple[int, int, int, int]]:
    """
    Compute ground intersection of two images with envelopes:
    1/ Create envelopes for left, right images
    2/ Read vectors and polygons with adequate EPSG codes.
    3/ compute the ground polygon of the intersection of two envelopes
    4/ Write the GPKG vector

    Returns a shapely polygon and intersection bounding box

    :raise: Exception when the envelopes don't intersect one to each other

    :param conf: cars input configuration dictionary
    :param geometry_loader_to_use: name of geometry loader to use
    :param shp1_path: Path to the output shapefile left
    :param shp2_path: Path to the output shapefile right
    :param dem_dir: Directory containing DEM tiles
    :param default_alt: Default altitude above ellipsoid
    :param out_intersect_path: out vector file path to create
    :return: a tuple with the shapely polygon of the intersection
             and the intersection's bounding box
             (described by a tuple (minx, miny, maxx, maxy))
    """
    # Create left, right envelopes from images and dem, default_alt
    geo_loader = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            geometry_loader_to_use
        )
    )

    geo_loader.image_envelope(
        conf,
        input_parameters.PRODUCT1_KEY,
        shp1_path,
        dem=dem_dir,
        default_alt=default_alt,
        geoid=geoid,
    )
    geo_loader.image_envelope(
        conf,
        input_parameters.PRODUCT2_KEY,
        shp2_path,
        dem=dem_dir,
        default_alt=default_alt,
        geoid=geoid,
    )

    # Read vectors shapefiles
    poly1, epsg1 = inputs.read_vector(shp1_path)
    poly2, epsg2 = inputs.read_vector(shp2_path)

    # Find polygon intersection from left, right polygons
    inter_poly, (
        inter_xmin,
        inter_ymin,
        inter_xmax,
        inter_ymax,
    ) = ground_polygon_from_envelopes(poly1, poly2, epsg1, epsg2, epsg1)

    # Write intersection file vector from inter_poly
    outputs.write_vector([inter_poly], out_intersect_path, epsg1)

    return inter_poly, (inter_xmin, inter_ymin, inter_xmax, inter_ymax)


def project_coordinates_on_line(
    x_coord: Union[float, np.ndarray],
    y_coord: Union[float, np.ndarray],
    origin: Union[List[float], np.ndarray],
    vec: Union[List[float], np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Project coordinates (x,y) on a line starting from origin with a
    direction vector vec, and return the euclidean distances between
    projected points and origin.

    :param x_coord: scalar or vector of coordinates x
    :param y_coord: scalar or vector of coordinates x
    :param origin: coordinates of origin point for line
    :param vec: direction vector of line
    :return: vector of distances of projected points to origin
    """
    assert len(x_coord) == len(y_coord)
    assert len(origin) == 2
    assert len(vec) == 2

    vec_angle = math.atan2(vec[1], vec[0])
    point_angle = np.arctan2(y_coord - origin[1], x_coord - origin[0])
    proj_angle = point_angle - vec_angle
    dist_to_origin = np.sqrt(
        np.square(x_coord - origin[0]) + np.square(y_coord - origin[1])
    )

    return dist_to_origin * np.cos(proj_angle)


def get_time_ground_direction(
    conf,
    geometry_loader_to_use: str,
    product_key: str,
    x_loc: float = None,
    y_loc: float = None,
    y_offset: float = None,
    dem: str = None,
    geoid: str = None,
) -> np.ndarray:
    """
    For a given image, compute the direction of increasing acquisition
    time on ground.
    Done by two localizations at "y" and "y+y_offset" values.

    :param conf: cars input configuration dictionary
    :param geometry_loader_to_use: name of geometry loader to use
    :param product_key: input_parameters.PRODUCT1_KEY or
           input_parameters.PRODUCT2_KEY to identify which geometric model shall
           be taken to perform the method
    :param x_loc: x location in image for estimation (default=center)
    :param y_loc: y location in image for estimation (default=1/4)
    :param y_offset: y location in image for estimation (default=1/2)
    :param dem: DEM for direct localisation function
    :param geoid: path to geoid file
    :return: normalized direction vector as a numpy array
    """
    # Define x: image center, y: 1/4 of image,
    # y_offset: 3/4 of image if not defined
    img = conf[input_parameters.create_img_tag_from_product_key(product_key)]
    img_size_x, img_size_y = inputs.rasterio_get_size(img)
    if x_loc is None:
        x_loc = img_size_x / 2
    if y_loc is None:
        y_loc = img_size_y / 4
    if y_offset is None:
        y_offset = img_size_y / 2

    # Check x, y, y_offset to be in image
    assert x_loc >= 0
    assert x_loc <= img_size_x
    assert y_loc >= 0
    assert y_loc <= img_size_y
    assert y_offset > 0
    assert y_loc + y_offset <= img_size_y

    # Get coordinates of time direction vectors
    geometry_loader = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            geometry_loader_to_use
        )
    )
    lat1, lon1, __ = geometry_loader.direct_loc(
        conf, product_key, x_loc, y_loc, dem=dem, geoid=geoid
    )
    lat2, lon2, __ = geometry_loader.direct_loc(
        conf, product_key, x_loc, y_loc + y_offset, dem=dem, geoid=geoid
    )

    # Create and normalize the time direction vector
    vec = np.array([lon1 - lon2, lat1 - lat2])
    vec = vec / np.linalg.norm(vec)

    return vec


def display_angle(vec):
    """
    Display angle in degree from a vector x
    :param vec: vector to display
    :return: angle in degree
    """
    return 180 * math.atan2(vec[1], vec[0]) / math.pi


def acquisition_direction(
    conf, geometry_loader_to_use, dem: str
) -> Tuple[np.ndarray]:
    """
    Computes the mean acquisition of the input images pair

    :param conf: cars input configuration dictionary
    :param geometry_loader_to_use: name of geometry loader to use
    :param dem: path to the dem directory
    :return: a tuple composed of :

        - the mean acquisition direction as a numpy array
        - the acquisition direction of the first product in the configuration
          as a numpy array
        - the acquisition direction of the second product in the configuration
          as a numpy array
    """
    # TODO remove ? usused
    vec1 = get_time_ground_direction(
        conf, geometry_loader_to_use, input_parameters.PRODUCT1_KEY, dem=dem
    )
    vec2 = get_time_ground_direction(
        conf, geometry_loader_to_use, input_parameters.PRODUCT2_KEY, dem=dem
    )
    time_direction_vector = (vec1 + vec2) / 2

    logging.info(
        "Time direction average azimuth: "
        "{}° (img1: {}°, img2: {}°)".format(
            display_angle(time_direction_vector),
            display_angle(vec1),
            display_angle(vec2),
        )
    )

    return time_direction_vector, vec1, vec2


def get_ground_direction(
    conf,
    geometry_loader_to_use: str,
    product_key: str,
    x_coord: float = None,
    y_coord: float = None,
    z0_coord: float = None,
    z_coord: float = None,
    dem: str = None,
    geoid: str = None,
) -> np.ndarray:
    """
    For a given image (x,y) point, compute the direction vector to ground
    The function uses the direct localization operation and makes a z
    variation to get a ground direction vector.
    By default, (x,y) is put at image center and z0, z at RPC geometric
    model limits.

    :param conf: cars input configuration dictionary
    :param geometry_loader_to_use: name of geometry loader to use
    :param product_key: input_parameters.PRODUCT1_KEY or
           input_parameters.PRODUCT2_KEY to identify which geometric model shall
           be taken to perform the method
    :param x_coord: X Coordinate in input image sensor
    :param y_coord: Y Coordinate in input image sensor
    :param z0_coord: Z altitude reference coordinate
    :param z_coord: Z Altitude coordinate to take the image
    :param dem: path to the dem directory
    :param geoid: path to the geoid file
    :return: (lat0,lon0,alt0, lat,lon,alt) origin and end vector coordinates
    """
    # Define x, y in image center if not defined
    img = conf[input_parameters.create_img_tag_from_product_key(product_key)]
    img_size_x, img_size_y = inputs.rasterio_get_size(img)

    if x_coord is None:
        x_coord = img_size_x / 2
    if y_coord is None:
        y_coord = img_size_y / 2

    # Check x, y to be in image
    assert x_coord >= 0
    assert x_coord <= img_size_x
    assert y_coord >= 0
    assert y_coord <= img_size_y

    # Define z and z0 from img RPC constraints if not defined
    (min_alt, max_alt) = utils.get_elevation_range_from_metadata(img)
    if z0_coord is None:
        z0_coord = min_alt
    if z_coord is None:
        z_coord = max_alt

    # Check z0 and z to be in RPC constraints
    assert z0_coord >= min_alt
    assert z0_coord <= max_alt
    assert z_coord >= min_alt
    assert z_coord <= max_alt

    # Get origin vector coordinate with z0 altitude
    geometry_loader = (
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            geometry_loader_to_use
        )
    )
    lat0, lon0, alt0 = geometry_loader.direct_loc(
        conf,
        product_key,
        x_coord,
        y_coord,
        z_coord=z0_coord,
        dem=dem,
        geoid=geoid,
    )
    # Get end vector coordinate with z altitude
    lat, lon, alt = geometry_loader.direct_loc(
        conf,
        product_key,
        x_coord,
        y_coord,
        z_coord=z_coord,
        dem=dem,
        geoid=geoid,
    )

    return np.array([lat0, lon0, alt0, lat, lon, alt])


def get_ground_angles(
    conf,
    geometry_loader_to_use,
    geoid: str = None,
    x1_coord: float = None,
    y1_coord: float = None,
    z1_0_coord: float = None,
    z1_coord: float = None,
    x2_coord: float = None,
    y2_coord: float = None,
    z2_0_coord: float = None,
    z2_coord: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For a given image (x,y) point, compute the Azimuth angle,
    Elevation angle (not the altitude !) and Range from Ground z0 perspective
    for both stereo image (img1: left and img2: right)

    Calculate also the convergence angle between the two satellites from ground.

    The function use get_ground_direction function to have coordinates of
    ground direction vector and compute angles and range.

    Ref: Jeong, Jaehoon. (2017).
    IMAGING GEOMETRY AND POSITIONING ACCURACY OF DUAL SATELLITE STEREO IMAGES:
    A REVIEW. ISPRS Annals of Photogrammetry, Remote Sensing and Spatial
    Information Sciences.
    IV-2/W4. 235-242. 10.5194/isprs-annals-IV-2-W4-235-2017.

    Perspectives: get bisector  elevation (BIE), and asymmetry angle

    :param conf: cars input configuration dictionary
    :param geometry_loader_to_use: name of geometry loader to use
    :param x1_coord: X Coordinate in input left image1  sensor
    :param y1_coord: Y Coordinate in input left image1 sensor
    :param z1_0_coord: Left image1 Z altitude origin coordinate
           for ground direction vector
    :param z1_coord:  Left image1 Z altitude end coordinate
           for ground direction vector
    :param x2_coord: X Coordinate in input right image2 sensor
    :param y2_coord: Y Coordinate in input right image2 sensor
    :param z2_0_coord: Right image2 Z altitude origin coordinate
           for ground direction vector
    :param z2_coord: Right image2 Z altitude end coordinate
           for ground direction vector
    :return: Left Azimuth, Left Elevation Angle,
             Right Azimuth, Right Elevation Angle, Convergence Angle
    """
    # TODO remove ? unused

    # Get image1 <-> satellite vector from image2 metadata geometric model
    lat1_0, lon1_0, alt1_0, lat1, lon1, alt1 = get_ground_direction(
        conf,
        geometry_loader_to_use,
        input_parameters.PRODUCT1_KEY,
        x1_coord,
        y1_coord,
        z1_0_coord,
        z1_coord,
        geoid=geoid,
    )
    # Get East North Up vector for left image1
    x1_e, y1_n, y1_u = enu1 = geo_to_enu(
        lat1, lon1, alt1, lat1_0, lon1_0, alt1_0
    )
    # Convert vector to Azimuth, Elevation, Range (unused)
    az1, elev_angle1, __ = enu_to_aer(x1_e, y1_n, y1_u)

    # Get image2 <-> satellite vector from image2 metadata geometric model
    lat2_0, lon2_0, alt2_0, lat2, lon2, alt2 = get_ground_direction(
        conf,
        geometry_loader_to_use,
        input_parameters.PRODUCT2_KEY,
        x2_coord,
        y2_coord,
        z2_0_coord,
        z2_coord,
        geoid=geoid,
    )
    # Get East North Up vector for right image2
    x2_e, y2_n, y2_u = enu2 = geo_to_enu(
        lat2, lon2, alt2, lat2_0, lon2_0, alt2_0
    )
    # Convert ENU to Azimuth, Elevation, Range (unused)
    az2, elev_angle2, __ = enu_to_aer(x2_e, y2_n, y2_u)

    # Get convergence angle from two enu vectors.
    convergence_angle = np.degrees(utils.angle_vectors(enu1, enu2))

    return az1, elev_angle1, az2, elev_angle2, convergence_angle
