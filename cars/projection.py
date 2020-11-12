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
This module contains some purpose functions using polygons and data projections that do not fit in other modules
"""

# Standard imports
from typing import List, Tuple
import os
import logging
import numpy as np

# Third party imports
import pandas
import xarray as xr
import rasterio as rio
from rasterio.features import shapes
from shapely.geometry import shape, Polygon
from shapely.ops import transform
import pyproj
import osgeo
from osgeo import osr

# cars import
from cars import utils
from cars import constants as cst


def get_projected_bounding_box(poly: Polygon, poly_epsg: int, target_epsg: int,
                               min_elev: float, max_elev: float) -> List[int]:
    """
    Get the maximum bounding box of the projected polygon considering an elevation range.
    To do so the polygon is projected two times using the min and max elevations. The bounding box containing the two
    projected polygons is then returned.

    :param poly: polygon to project
    :param poly_epsg: polygon epsg code
    :param target_epsg:  epsg code of the target referential
    :param min_elev: minimum elevation in the considering polygon
    :param max_elev: maximum elevation in the considering polygon
    :return: the maximum bounding box in the target epsg code referential as a list [xmin, ymin, xmax, ymax]
    """
    # construct two polygons from the initial one and add min and max elevations in each of them
    poly_pts_with_min_alt = []
    poly_pts_with_max_alt = []
    for pt in list(poly.exterior.coords):
        poly_pts_with_min_alt.append((pt[0], pt[1], min_elev))
        poly_pts_with_max_alt.append((pt[0], pt[1], max_elev))

    poly_elev_min = Polygon(poly_pts_with_min_alt)
    poly_elev_max = Polygon(poly_pts_with_max_alt)

    # project the polygons
    poly_elev_min = polygon_projection(poly_elev_min, poly_epsg, target_epsg)
    poly_elev_max = polygon_projection(poly_elev_max, poly_epsg, target_epsg)

    # retrieve the largest bounding box
    (xmin_poly_elev_min, ymin_poly_elev_min, xmax_poly_elev_min, ymax_poly_elev_min) = poly_elev_min.bounds
    (xmin_poly_elev_max, ymin_poly_elev_max, xmax_poly_elev_max, ymax_poly_elev_max) = poly_elev_max.bounds

    xmin = min(xmin_poly_elev_min, xmin_poly_elev_max)
    ymin = min(ymin_poly_elev_min, ymin_poly_elev_max)
    xmax = max(xmax_poly_elev_min, xmax_poly_elev_max)
    ymax = max(ymax_poly_elev_min, ymax_poly_elev_max)

    return [xmin, ymin, xmax, ymax]


def compute_dem_intersection_with_poly(srtm_dir, ref_poly, ref_epsg):
    """
    Compute the intersection polygon between the defined dem regions and the reference polygon in input

    :raise Exception when the input dem does not intersect the reference polygon

    :param srtm_dir: srtm directory
    :type srtm_dir: str
    :param ref_poly: reference polygon
    :type ref_poly: Polygon
    :param ref_epsg: reference epsg code
    :type ref_epsg: int
    :return: The intersection polygon between the defined dem regions and the reference polygon in input
    :rtype Polygon
    """
    dem_poly = None
    for r, d, f in os.walk(srtm_dir):
        logging.info(
            'Browsing all files of the srtm dir. '
            'Some files might be unreadable by rasterio (non blocking matter).')
        for file in f:
            unsupported_formats = ['.omd']
            _, ext = os.path.splitext(file)
            if ext not in unsupported_formats:
                if utils.rasterio_can_open(os.path.join(srtm_dir, file)):
                    with rio.open(os.path.join(srtm_dir, file)) as data:

                        xmin = min(data.bounds.left, data.bounds.right)
                        ymin = min(data.bounds.bottom, data.bounds.top)
                        xmax = max(data.bounds.left, data.bounds.right)
                        ymax = max(data.bounds.bottom, data.bounds.top)

                        try:
                            file_epsg = data.crs.to_epsg()
                            file_bb = Polygon(
                                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)])

                            # transform polygon if needed
                            if ref_epsg != file_epsg:
                                file_bb = polygon_projection(
                                    file_bb, file_epsg, ref_epsg)

                            # if the srtm tile intersects the reference polygon
                            if file_bb.intersects(ref_poly):
                                local_dem_poly = None

                                # retrieve valid polygons
                                for poly, val in shapes(
                                        data.dataset_mask(), transform=data.transform):
                                    if val != 0:
                                        poly = shape(poly)
                                        poly = poly.buffer(0)
                                        if ref_epsg != file_epsg:
                                            poly = polygon_projection(
                                                poly, file_epsg, ref_epsg)

                                        # combine valid polygons
                                        if local_dem_poly is None:
                                            local_dem_poly = poly
                                        else:
                                            local_dem_poly = poly.union(
                                                local_dem_poly)

                                # combine the tile valid polygon to the other
                                # tiles' ones
                                if dem_poly is None:
                                    dem_poly = local_dem_poly
                                else:
                                    dem_poly = dem_poly.union(local_dem_poly)

                        except AttributeError as e:
                            logging.error(
                                'Impossible to read the SRTM tile epsg code: {}'.format(e))

    # compute dem coverage polygon over the reference polygon
    if dem_poly is None or not dem_poly.intersects(ref_poly):
        raise Exception('The input DEM does not intersect the useful zone')

    dem_cover = dem_poly.intersection(ref_poly)

    area_cover = dem_cover.area
    area_inter = ref_poly.area

    return dem_cover, area_cover / area_inter * 100.0


def polygon_projection(poly, from_epsg, to_epsg):
    """
    Projects a polygon from an initial epsg code to another

    :param poly: poly to project
    :type poly: Polygon
    :param from_epsg: initial epsg code
    :type from_epsg: int
    :param to_epsg: final epsg code
    :type to_epsg: int
    :return: The polygon in the final projection
    :rtype: Polygon
    """
    project = pyproj.Transformer.from_proj(
        pyproj.Proj(
            init='epsg:{}'.format(from_epsg)), pyproj.Proj(
            init='epsg:{}'.format(to_epsg)))
    poly = transform(project.transform, poly)

    return poly


def geo_to_ecef(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray)\
                -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Point transformation from Geodetic of ellipsoid WGS-84) to ECEF
    ECEF : Earth-centered, Earth-fixed

    Python code solution without calling OSR through points_cloud_conversion :

    rad = np.float64(6378137.0)        # Radius of the Earth (in meters)
    f = np.float64(1.0/298.257223563)  # Flattening factor WGS84 Model
    cos_lat = np.cos(lat * np.pi / 180)
    sin_lat = np.sin(lat * np.pi / 180)
    cos_long = np.cos(lon * np.pi / 180)
    sin_long = np.sin(lon * np.pi / 180)
    c = 1 / np.sqrt(cos_lat * cos_lat + (1 - f) * (1 - f) * sin_lat * sin_lat)
    s = (1 - f) * (1 - f) * c
    x = (rad*c + altitude) * cos_lat * cos_long
    y = (rad*c + altitude) * cos_lat * sin_long
    z = (rad*s + altitude) * sin_lat
    return x, y, z

    :param lat : input geodetic latitude (angle in degree)
    :param lon :  input geodetic longitude (angle in degree)
    :param alt : input altitude above geodetic ellipsoid (meters)
    :return:  ECEF (Earth centered, Earth fixed) x, y, z coordinates tuple (meters)
    """
    epsg_in=4979 # EPSG code for Geocentric WGS84 in lat, lon, alt (degree)
    epsg_out=4978 # EPSG code for ECEF WGS84 in x, y, z (meters)
    x, y, z = points_cloud_conversion([(lon, lat, alt)], epsg_in, epsg_out)[0]

    return x, y, z

def ecef_to_enu(x: np.ndarray, y: np.ndarray,  z: np.ndarray,\
                lat0: np.ndarray, lon0: np.ndarray,  alt0: np.ndarray)\
                -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Coordinates conversion from ECEF Earth Centered to
    East North Up Coordinate from a reference point (lat0, lon0, alt0)

    TODO : Use OSR when ENU will be available.

    See Wikipedia page for details :
    https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU

    :param x : target x ECEF coordinate (meters)
    :param y : target y ECEF coordinate (meters)
    :param z : target z ECEF coordinate (meters)
    :param lat0: Reference geodetic latitude
    :param lon0: Reference geodetic longitude
    :param alt0: Reference altitude above geodetic ellipsoid (meters)
    :return: ENU (xEast, yNorth zUp) target coordinates tuple (meters)
    """
    rad = np.float64(6378137.0)        # Radius of the Earth (in meters)
    f = np.float64(1.0/298.257223563)  # Flattening factor WGS84 Model
    e = (1 - f) * (1 - f)
    e2 = 1 - e

    cos_lat0 = np.cos(np.radians(lat0))
    sin_lat0 = np.sin(np.radians(lat0))

    cos_long0 = np.cos(np.radians(lon0))
    sin_long0 = np.sin(np.radians(lon0))

    c0 = 1 / np.sqrt(cos_lat0 * cos_lat0  + e * sin_lat0 * sin_lat0)

    # Determine ECEF coordinates from reference geodetic
    x0, y0, z0 = geo_to_ecef(lat0, lon0, alt0)
    # x0 = (rad*c0 + alt0) * cos_lat0 * cos_long0
    # y0 = (rad*c0 + alt0) * cos_lat0 * sin_long0
    # z0 = (rad*c0*(1-e2) + alt0) * sin_lat0

    xEast = (-(x-x0) * sin_long0) + ((y-y0)*cos_long0)
    yNorth = (-cos_long0*sin_lat0*(x-x0)) - (sin_lat0*sin_long0*(y-y0)) + (cos_lat0*(z-z0))
    zUp = (cos_lat0*cos_long0*(x-x0)) + (cos_lat0*sin_long0*(y-y0)) + (sin_lat0*(z-z0))

    return xEast, yNorth, zUp

def geo_to_enu(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
               lat0: np.ndarray, lon0: np.ndarray, alt0: np.ndarray)\
               -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    point transformation from WGS-84 Geodetic coordinates to to ENU
    Use geo_to_ecef and ecef_to_enu functions

    :param lat : input geodetic latitude (angle in degree)
    :param lon :  input geodetic longitude (angle in degree)
    :param alt : input altitude above geodetic ellipsoid (meters)
    :param lat0: Reference geodetic latitude
    :param lon0: Reference geodetic longitude
    :param alt0: Reference altitude above geodetic ellipsoid (meters)
    :return: ENU (xEast, yNorth zUp) target coordinates tuple (meters)
    """
    x, y, z = geo_to_ecef(lat, lon, alt)
    return ecef_to_enu(x, y, z, lat0, lon0, alt0)

def enu_to_aer(e: np.ndarray, n: np.ndarray, u: np.ndarray)\
               -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ENU coordinates to Azimuth, Elevation angle, Range from ENU origin
    Beware : Elevation angle is different from altitude.
    :param e: ENU East coordinate (meters)
    :param n: ENU North coordinate (meters)
    :param u: ENU Up coordinate (meters)
    :return: Azimuth, Elevation Angle, Slant Range tuple (degrees, degres, meters)
    """

    r = np.hypot(e, n)
    slantRange = np.hypot(r, u) # Distance of e, n, u vector
    elev = np.arctan2(u, r)
    az = np.arctan2(e, n) % (2 * np.pi) # From [-pi,+pi] to [0,2pi]

    az = np.degrees(az)
    elev = np.degrees(elev)

    return az, elev, slantRange

def geo_to_aer(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
               lat0: np.ndarray, lon0: np.ndarray, alt0: np.ndarray)\
               -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gives Azimuth, Elevation angle and Slant Range from a Reference to a Point with geodetic coordinates.
    :param lat : input geodetic latitude (angle in degree)
    :param lon : input geodetic longitude (angle in degree)
    :param alt : input altitude above geodetic ellipsoid (meters)
    :param lat0: Reference geodetic latitude
    :param lon0: Reference geodetic longitude
    :param alt0: Reference altitude above geodetic ellipsoid (meters)
    :return: Azimuth, Elevation Angle, Slant Range tuple (degrees, degres, meters)
    """
    e, n, u = geo_to_enu(lat, lon, alt, lat0, lon0, alt0)
    return enu_to_aer(e, n, u)

def points_cloud_conversion(cloud_in, epsg_in, epsg_out):
    """
    Convert a point cloud from a SRS to another one.

    :param cloud_in: cloud to project
    :type cloud_in: numpy array
    :param epsg_in: EPSG code of the input SRS
    :type epsg_in: int
    :param epsg_out: EPSG code of the ouptut SRS
    :type epsg_out: int
    :returns: Projected point cloud
    :rtype: numpy array
    """
    srs_in = osr.SpatialReference()
    srs_in.ImportFromEPSG(epsg_in)
    srs_out = osr.SpatialReference()
    srs_out.ImportFromEPSG(epsg_out)
    # GDAL 3.0 Coordinate transformation (backwards compatibility)
    # https://github.com/OSGeo/gdal/issues/1546
    if int(osgeo.__version__[0]) >= 3:
        srs_in.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        srs_out.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    conversion = osr.CoordinateTransformation(srs_in, srs_out)
    cloud_in = conversion.TransformPoints(cloud_in)
    cloud_in = np.array(cloud_in)

    return cloud_in


def get_xyz_np_array_from_dataset(cloud_in: xr.Dataset) -> Tuple[np.array, List[int]]:
    """
    Get a numpy array of size (nb_points, 3) with the columns being the x, y and z coordinates from a dataset as given
    in output of the triangulation.
    The original epipolar geometry shape is also given in output in order to reshape the output numpy array in its
    original geometry if necessary.

    :param cloud_in: input xarray dataset
    :return: a tuple composed of the xyz numàpy array and its original shape
    """
    xyz = np.stack(
        (cloud_in[cst.X].values,
         cloud_in[cst.Y].values,
         cloud_in[cst.Z]),
        axis=-1)
    xyz_shape = xyz.shape
    xyz = np.reshape(xyz, (-1, 3))

    return xyz, xyz_shape


def get_converted_xy_np_arrays_from_dataset(cloud_in: xr.Dataset, epsg_out: int) -> Tuple[np.array, np.array]:
    """
    Get the x and y coordinates as numpy array in the new referential indicated by epsg_out.

    :param cloud_in: input xarray dataset
    :param epsg_out: target epsg code
    :return: a tuple composed of the x and y numpy arrays
    """
    xyz, xyz_shape = get_xyz_np_array_from_dataset(cloud_in)

    xyz = points_cloud_conversion(xyz, int(cloud_in.attrs[cst.EPSG]), epsg_out)
    xyz = xyz.reshape(xyz_shape)
    proj_x = xyz[:, :, 0]
    proj_y = xyz[:, :, 1]

    return proj_x, proj_y


def points_cloud_conversion_dataset(cloud: xr.Dataset, epsg_out: int):
    """
    Convert a point cloud as an xarray.Dataset to another epsg (inplace)

    :param cloud: cloud to project
    :param epsg_out: EPSG code of the ouptut SRS
    """

    if cloud.attrs[cst.EPSG] != epsg_out:

        xyz, xyz_shape = get_xyz_np_array_from_dataset(cloud)

        xyz = points_cloud_conversion(xyz, int(cloud.attrs[cst.EPSG]), epsg_out)
        xyz = xyz.reshape(xyz_shape)

        # Update cloud_in x, y and z values
        cloud[cst.X].values = xyz[:, :, 0]
        cloud[cst.Y].values = xyz[:, :, 1]
        cloud[cst.Z].values = xyz[:, :, 2]

        # Update EPSG code
        cloud.attrs[cst.EPSG] = epsg_out


def points_cloud_conversion_dataframe(cloud: pandas.DataFrame, epsg_in: int, epsg_out: int):
    """
    Convert a point cloud as a panda.DataFrame to another epsg (inplace)

    :param cloud: cloud to project
    :param epsg_in: EPSG code of the input SRS
    :param epsg_out: EPSG code of the ouptut SRS
    """
    xyz_in = cloud.loc[:, [cst.X, cst.Y, cst.Z]].values

    if xyz_in.shape[0] != 0:
        xyz_in = points_cloud_conversion(xyz_in, epsg_in, epsg_out)
        cloud[cst.X] = xyz_in[:, 0]
        cloud[cst.Y] = xyz_in[:, 1]
        cloud[cst.Z] = xyz_in[:, 2]
