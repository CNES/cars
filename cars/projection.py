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
from osgeo import osr

# cars import
from cars import utils


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
    conversion = osr.CoordinateTransformation(srs_in, srs_out)
    cloud_in = conversion.TransformPoints(cloud_in)
    cloud_in = np.array(cloud_in)

    #if epsg_out!=4326:
    #    cloud_out = [np.float32(p) for p in cloud_out]

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
        (cloud_in['x'].values,
         cloud_in['y'].values,
         cloud_in['z']),
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

    xyz = points_cloud_conversion(xyz, int(cloud_in.attrs['epsg']), epsg_out)
    xyz = xyz.reshape(xyz_shape)
    proj_x = xyz[:, :, 0]
    proj_y = xyz[:, :, 1]

    return proj_x, proj_y


def points_cloud_conversion_dataset(cloud_in, epsg_out):
    """
    Convert a point cloud as an xarray.Dataset to another epsg

    :param cloud_in: cloud to project
    :type cloud_in: numpy array
    :param epsg_out: EPSG code of the ouptut SRS
    :type epsg_out: int
    :returns: Projected point cloud
    :rtype: numpy array
    """

    cloud_out = cloud_in

    if cloud_in.attrs['epsg'] != epsg_out:

        xyz, xyz_shape = get_xyz_np_array_from_dataset(cloud_in)

        xyz = points_cloud_conversion(xyz, int(cloud_in.attrs['epsg']), epsg_out)
        xyz = xyz.reshape(xyz_shape)

        values = {
            'x': (['row', 'col'], xyz[:, :, 0]),
            'y': (['row', 'col'], xyz[:, :, 1]),
            'z': (['row', 'col'], xyz[:, :, 2]),
            'pandora_msk': (['row', 'col'], cloud_in['pandora_msk'].values)
        }
        values_list = [key for key, _ in cloud_in.items()]
        if 'msk' in values_list:
            values['msk'] = (['row', 'col'], cloud_in['msk'].values)

        # Copy attributes
        cloud_out = xr.Dataset(values,
                               coords=cloud_in.coords)

        for k, v in cloud_in.attrs.items():
            cloud_out.attrs[k] = v

        # Update EPSG code
        cloud_out.attrs['epsg'] = epsg_out

    return cloud_out


def points_cloud_conversion_dataframe(cloud: pandas.DataFrame, epsg_in: int, epsg_out: int):
    """
    Convert a point cloud as a panda.DataFrame to another epsg (inplace)

    :param cloud: cloud to project
    :param epsg_in: EPSG code of the input SRS
    :param epsg_out: EPSG code of the ouptut SRS
    """
    xyz_in = cloud.loc[:, ['x', 'y', 'z']].values

    if xyz_in.shape[0] != 0:
        xyz_in = points_cloud_conversion(xyz_in, epsg_in, epsg_out)
        cloud.x = xyz_in[:, 0]
        cloud.y = xyz_in[:, 1]
        cloud.z = xyz_in[:, 2]
