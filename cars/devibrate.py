#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
cars-devibrate: devibrate a high resolution DSM using a low resolution DSM
"""

import argparse
import json
import logging
import math

# Standard imports
import os
import pickle
from typing import List, Union

# Third party imports
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
import xarray as xr
from rasterio.windows import bounds as to_bounds
from rasterio.windows import from_bounds
from scipy import interpolate
from scipy.signal import butter, filtfilt, lfilter, lfilter_zi

# CARS / SHARELOC imports
from shareloc.dtm_reader import interpolate_geoid_height
from shareloc.geofunctions import triangulation

from cars.applications.rasterization import rasterization_algo as rasterization
from cars.core.geometry.abstract_geometry import AbstractGeometry
from cars.core.geometry.shareloc_geometry import SharelocGeometry

# Get full path geoid
package_path = os.path.dirname(__file__)
GEOID_DEFAULT = os.path.join(package_path, "conf", "geoid/egm96.grd")


def acquisition_direction(
    sensor1, geomodel1, sensor2, geomodel2, geometry_plugin
):
    """
    Computes the mean acquisition of the input images pair

    :param sensor1: sensor image name of the first product
    :param geomodel1: geomodel name of the first product
    :param sensor2: sensor image name of the second product
    :param geomodel2: geomodel name of the second product
    :return: a tuple composed of :
        - the mean acquisition direction as a numpy array
        - the acquisition direction of the first product as a numpy array
        - the acquisition direction of the second product as a numpy array
    """
    vec1 = get_time_ground_direction(sensor1, geomodel1, geometry_plugin)
    vec2 = get_time_ground_direction(sensor2, geomodel2, geometry_plugin)
    time_direction_vector = (vec1 + vec2) / 2

    def display_angle(vec):
        """
        Display angle in degree from a vector x
        :param vec: vector to display
        :return: angle in degree
        """
        return 180 * math.atan2(vec[1], vec[0]) / math.pi

    logging.info(
        "Time direction average azimuth: "
        "{}deg (img1: {}deg, img2: {}deg)".format(
            display_angle(time_direction_vector),
            display_angle(vec1),
            display_angle(vec2),
        )
    )

    return time_direction_vector, vec1, vec2


def get_time_ground_direction(
    sensor,
    geomodel,
    geometry_plugin,
    x_loc: float = None,
    y_loc: float = None,
    y_offset: float = None,
) -> np.ndarray:
    """
    For a given image, compute the direction of increasing acquisition
    time on ground.
    Done by two localizations at "y" and "y+y_offset" values.

    :param sensor: sensor image name
    :param geomodel: geomodel name
    :param x_loc: x location in image for estimation (default=center)
    :param y_loc: y location in image for estimation (default=1/4)
    :param y_offset: y location in image for estimation (default=1/2)
    :param dem: DEM for direct localisation function
    :param geoid: path to geoid file
    :return: normalized direction vector as a numpy array
    """
    # Define x: image center, y: 1/4 of image,
    # y_offset: 3/4 of image if not defined
    with rio.open(sensor) as src:
        img_size_x, img_size_y = src.width, src.height
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
    lat1, lon1, __ = geometry_plugin.direct_loc(sensor, geomodel, x_loc, y_loc)
    lat2, lon2, __ = geometry_plugin.direct_loc(
        sensor, geomodel, x_loc, y_loc + y_offset
    )

    # Create and normalize the time direction vector
    vec = np.array([lon1 - lon2, lat1 - lat2])
    vec = vec / np.linalg.norm(vec)

    return vec


def project_coordinates_on_line(
    x_coord: Union[float, np.ndarray],
    y_coord: Union[float, np.ndarray],
    origin: Union[List[float], np.ndarray],
    vec: Union[List[float], np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Project coordinates (x, y) on a line starting from origin with a
    direction vector vec, and return the euclidean distances between
    projected points and origin.

    :param x_coord: scalar or vector of coordinates x
    :param y_coord: scalar or vector of coordinates y
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


def lowres_initial_dem_splines_fit(
    lowres_dsm_from_matches: xr.Dataset,
    lowres_initial_dem: xr.Dataset,
    origin: np.ndarray,
    time_direction_vector: np.ndarray,
    ext: int = 3,
    order: int = 3,
    min_pts_per_time: int = 100,
    min_pts_along_time_direction: int = 100,
    butterworth_filter_order: int = 3,
    butterworth_critical_frequency: float = 0.05,
):
    """
    This function takes 2 datasets containing DSM and models the
    difference between the two as an UnivariateSpline along the
    direction given by origin and time_direction_vector. Internally,
    it looks for the highest smoothing factor that satisfies the rmse threshold.

    :param lowres_dsm_from_matches: Dataset containing the low resolution DSM
        obtained from matches, as returned by the
        rasterization.simple_rasterization_dataset function.
    :param lowres_initial_dem: Dataset containing the low resolution DEM,
        on the same grid as lowres_dsm_from_matches
    :param origin: coordinates of origin point for line
    :type origin: list(float) or np.array(float) of size 2
    :param time_direction_vector: direction vector of line
    :type time_direction_vector: list(float) or np.array(float) of size 2
    :param ext: behavior outside of interpolation domain
    :param order: spline order
    :param min_pts_per_time: minimum number of points for
        each measurement
    :param min_pts_along_time_direction:  minimum number of points for
        time direction
    :param butterworth_filter_order: Order of the filter.
        See scipy.signal.butter
    :param butterworth_critical_frequency: The filter critical frequency
        or frequencies. See scipy.signal.butter
    """
    # Initial DSM difference
    dsm_diff = lowres_initial_dem.hgt - lowres_dsm_from_matches.hgt

    # Form arrays of coordinates
    x_values_2d, y_values_2d = np.meshgrid(dsm_diff.x, dsm_diff.y)

    # Project coordinates on the line
    # of increasing acquisition time to form 1D coordinates
    # If there are sensor oscillations, they will occur in this direction
    linear_coords = project_coordinates_on_line(
        x_values_2d, y_values_2d, origin, time_direction_vector
    )

    # Form a 1D array with diff values indexed with linear coords
    linear_diff_array = xr.DataArray(
        dsm_diff.values.ravel(), coords={"l": linear_coords.ravel()}, dims=("l")
    )
    linear_diff_array = linear_diff_array.dropna(dim="l")
    linear_diff_array = linear_diff_array.sortby("l")

    # Apply median filtering within cell matching input dsm resolution
    min_l = np.min(linear_diff_array.l)
    max_l = np.max(linear_diff_array.l)
    nbins = int(
        math.ceil((max_l - min_l) / (lowres_dsm_from_matches.resolution))
    )
    median_linear_diff_array = linear_diff_array.groupby_bins(
        "l", nbins
    ).median()
    median_linear_diff_array = median_linear_diff_array.rename({"l_bins": "l"})
    median_linear_diff_array = median_linear_diff_array.assign_coords(
        {"l": np.array([d.mid for d in median_linear_diff_array.l.data])}
    )

    count_linear_diff_array = linear_diff_array.groupby_bins("l", nbins).count()
    count_linear_diff_array = count_linear_diff_array.rename({"l_bins": "l"})
    count_linear_diff_array = count_linear_diff_array.assign_coords(
        {"l": np.array([d.mid for d in count_linear_diff_array.l.data])}
    )

    # Filter measurements with insufficient amount of points
    median_linear_diff_array = median_linear_diff_array.where(
        count_linear_diff_array > min_pts_per_time
    ).dropna(dim="l")

    if len(median_linear_diff_array) < min_pts_along_time_direction:
        raise RuntimeError(
            "Insufficient amount of points ({} < {}) along time direction "
            "after measurements filtering to estimate correction "
            "to fit initial DEM".format(
                len(median_linear_diff_array), min_pts_along_time_direction
            )
        )

    # Apply butterworth lowpass filter to retrieve only the low frequency
    # (from example of doc: https://docs.scipy.org/doc/scipy/reference/
    # generated/scipy.signal.butter.html#scipy.signal.butter )
    b, a = butter(butterworth_filter_order, butterworth_critical_frequency)
    zi_filter = lfilter_zi(b, a)
    z_filter, _ = lfilter(
        b,
        a,
        median_linear_diff_array.values,
        zi=zi_filter * median_linear_diff_array.values[0],
    )
    lfilter(b, a, z_filter, zi=zi_filter * z_filter[0])
    filtered_median_linear_diff_array = xr.DataArray(
        filtfilt(b, a, median_linear_diff_array.values),
        coords=median_linear_diff_array.coords,
    )

    # Estimate performances of spline s = 100 * length of diff array
    smoothing_factor = 100 * len(filtered_median_linear_diff_array.l)
    splines = interpolate.UnivariateSpline(
        filtered_median_linear_diff_array.l,
        filtered_median_linear_diff_array.values,
        ext=ext,
        k=order,
        s=smoothing_factor,
    )
    estimated_correction = xr.DataArray(
        splines(filtered_median_linear_diff_array.l),
        coords=filtered_median_linear_diff_array.coords,
    )
    rmse = (filtered_median_linear_diff_array - estimated_correction).std(
        dim="l"
    )

    target_rmse = 0.3

    # If RMSE is too high, try to decrease smoothness until it fits
    while rmse > target_rmse and smoothing_factor > 0.001:
        # divide s by 2
        smoothing_factor /= 2

        # Estimate splines
        splines = interpolate.UnivariateSpline(
            filtered_median_linear_diff_array.l,
            filtered_median_linear_diff_array.values,
            ext=ext,
            k=order,
            s=smoothing_factor,
        )

        # Compute RMSE
        estimated_correction = xr.DataArray(
            splines(filtered_median_linear_diff_array.l),
            coords=filtered_median_linear_diff_array.coords,
        )

        rmse = (filtered_median_linear_diff_array - estimated_correction).std(
            dim="l"
        )

    logging.debug(
        "Best smoothing factor for splines regression: "
        "{} (rmse={})".format(smoothing_factor, rmse)
    )

    # Return estimated spline
    return splines


def read_lowres_dsm(srtm_path, startx, starty, endx, endy):
    """
    Read an extract of the low resolution input DSM and return it as an Array
    """

    with rio.open(srtm_path) as src:
        window = from_bounds(startx, starty, endx, endy, src.transform)
        window = window.round_offsets()
        window = window.round_lengths()
        bounds = to_bounds(window, src.transform)
        resolution = src.res[0]
        dem_as_array = src.read(1, window=window)

    newstartx, newstarty = bounds[0], bounds[-1]
    sizex, sizey = window.width, window.height

    x_values_1d = np.linspace(
        newstartx + 0.5 * resolution,
        newstartx + resolution * (sizex + 0.5),
        sizex,
        endpoint=False,
    )

    y_values_1d = np.linspace(
        newstarty - 0.5 * resolution,
        newstarty - resolution * (sizey + 0.5),
        sizey,
        endpoint=False,
    )

    dims = ["y", "x"]
    coords = {"x": x_values_1d, "y": y_values_1d}
    dsm_as_ds = xr.Dataset({"hgt": (dims, dem_as_array)}, coords=coords)
    dsm_as_ds["epsg"] = 4326
    dsm_as_ds["resolution"] = resolution
    return dsm_as_ds, newstartx, newstarty, sizex, sizey, resolution


def compute_splines(
    sensor1,
    geomodel1,
    sensor2,
    geomodel2,
    matches,
    srtm_path,
    geoid_path,
    out_dir,
    min_pts_per_time: int = 100,
    min_pts_along_time_direction: int = 100,
    butterworth_filter_order: int = 3,
    butterworth_critical_frequency: float = 0.05,
):
    """
    Compute a spline dict containing estimated splines, origin
    and time_direction_vector
    """
    geometry_plugin = AbstractGeometry(  # pylint: disable=E0110
        "SharelocGeometry"
    )

    # retrieve time direction from models
    time_direction_vector, _, _ = acquisition_direction(
        sensor1, geomodel1, sensor2, geomodel2, geometry_plugin
    )

    # load matches and triangulate them
    corrected_matches = np.load(matches)
    disp = corrected_matches[:, 0] - corrected_matches[:, 2]
    mini = np.percentile(disp, 5.0)
    maxi = np.percentile(disp, 95)
    corrected_matches = corrected_matches[
        np.logical_and(disp >= mini, disp <= maxi), :
    ]

    shareloc_model1 = SharelocGeometry.load_geom_model(geomodel1)
    shareloc_model2 = SharelocGeometry.load_geom_model(geomodel2)

    matches_sensor = corrected_matches[:, 4:8]
    [__, pc_from_matches, __] = triangulation.sensor_triangulation(
        matches_sensor, shareloc_model1, shareloc_model2
    )

    positions = np.array(pc_from_matches[:, 0:2])
    geoid_height = interpolate_geoid_height(geoid_path, positions)
    pc_from_matches[:, 2] = pc_from_matches[:, 2] - geoid_height

    # deduce area from sift
    pc_xx = pc_from_matches[:, 0]
    pc_yy = pc_from_matches[:, 1]
    startx, endx = pc_xx.min(), pc_xx.max()
    starty, endy = pc_yy.min(), pc_yy.max()

    # read corresponding dem
    lowres_initial_dem, startx, starty, sizex, sizey, resolution = (
        read_lowres_dsm(srtm_path, startx, starty, endx, endy)
    )

    points_cloud_from_matches = pd.DataFrame(
        data=pc_from_matches, columns=["x", "y", "z"]
    )

    # rasterize point cloud (superimposable image with lowres initial dem)
    points_cloud_from_matches.attrs["attributes"] = {"number_of_pc": 0}
    lowres_dsm = rasterization.simple_rasterization_dataset_wrapper(
        points_cloud_from_matches,
        resolution,
        4326,
        xstart=startx,
        ystart=starty,
        xsize=sizex,
        ysize=sizey,
    )

    lowres_dsm_diff = lowres_initial_dem - lowres_dsm
    origin = [
        float(lowres_dsm_diff["x"][0].values),
        float(lowres_dsm_diff["y"][0].values),
    ]

    # fit initial dem and low res dsm from sift and deduce splines correction
    splines = lowres_initial_dem_splines_fit(
        lowres_dsm,
        lowres_initial_dem,
        origin,
        time_direction_vector,
        min_pts_per_time=min_pts_per_time,
        min_pts_along_time_direction=min_pts_along_time_direction,
        butterworth_filter_order=butterworth_filter_order,
        butterworth_critical_frequency=butterworth_critical_frequency,
    )

    # save intermediate data
    lowres_initial_dem_file = os.path.join(out_dir, "lowres_initial_dem.nc")
    lowres_initial_dem.to_netcdf(lowres_initial_dem_file)
    lowres_dsm_file = os.path.join(out_dir, "lowres_dsm.nc")
    lowres_dsm.to_netcdf(lowres_dsm_file)
    lowres_dsm_diff_file = os.path.join(out_dir, "lowres_dsm_diff.nc")
    lowres_dsm_diff.to_netcdf(lowres_dsm_diff_file)

    # use splines for correction introduced in rasterization
    lowres_dsm_as_df = lowres_dsm.to_dataframe().reset_index()

    distance_vector = project_coordinates_on_line(
        lowres_dsm_as_df["x"],
        lowres_dsm_as_df["y"],
        origin,
        time_direction_vector,
    )
    correction = splines(distance_vector)

    lowres_dsm["hgt"] = lowres_dsm["hgt"] + correction.reshape(
        lowres_dsm["hgt"].shape
    )
    new_lowres_dsm_file = os.path.join(out_dir, "new_lowres_dsm.nc")
    lowres_dsm.to_netcdf(new_lowres_dsm_file)
    new_lowres_dsm_diff = lowres_initial_dem - lowres_dsm
    new_lowres_dsm_diff_file = os.path.join(out_dir, "new_lowres_dsm_diff.nc")
    new_lowres_dsm_diff.to_netcdf(new_lowres_dsm_diff_file)

    return {
        "splines": splines,
        "origin": origin,
        "time_direction_vector": time_direction_vector,
    }


def cars_devibrate(
    used_conf,
    srtm_path,
    geoid_path,
    min_pts_per_time: int = 100,
    min_pts_along_time_direction: int = 100,
    butterworth_filter_order: int = 3,
    butterworth_critical_frequency: float = 0.05,
):
    """
    Main fonction. Expects a dictionary from the CLI
    or directly the input parameters.
    """
    out_dir = os.path.dirname(used_conf)

    with open(used_conf, "r", encoding="utf8") as jsonfile:
        data = json.load(jsonfile)

    pairing = data["inputs"]["pairing"]
    assert len(pairing) == 1

    sensor1 = data["inputs"]["sensors"][pairing[0][0]]["image"]
    geomodel1 = data["inputs"]["sensors"][pairing[0][0]]["geomodel"]
    sensor2 = data["inputs"]["sensors"][pairing[0][1]]["image"]
    geomodel2 = data["inputs"]["sensors"][pairing[0][1]]["geomodel"]

    matches = os.path.join(
        out_dir,
        "dump_dir",
        "sparse_matching.sift",
        "_".join(pairing[0]),
        "filtered_matches.npy",
    )
    if not os.path.isfile(matches):
        raise RuntimeError(
            "Matches must be saved: Set applications.sparse_matching."
            "sift.save_intermediate_data to true in CARS configuration file."
        )

    dsm_path = os.path.join(out_dir, "dsm", "dsm.tif")
    if not os.path.isfile(dsm_path):
        raise RuntimeError("DSM must be generated: set product level to `dsm`")
    splines_path = os.path.join(out_dir, "splines.pck")

    if os.path.exists(splines_path) is False:
        splines_dict = compute_splines(
            sensor1,
            geomodel1,
            sensor2,
            geomodel2,
            matches,
            srtm_path,
            geoid_path,
            out_dir,
            min_pts_per_time=min_pts_per_time,
            min_pts_along_time_direction=min_pts_along_time_direction,
            butterworth_filter_order=butterworth_filter_order,
            butterworth_critical_frequency=butterworth_critical_frequency,
        )

        with open(splines_path, "wb") as writer:
            pickle.dump(splines_dict, writer)
    else:
        with open(splines_path, "rb") as reader:
            splines_dict = pickle.load(reader)

    with rio.open(dsm_path) as src:
        bounds = src.bounds
        startx, starty = bounds[0], bounds[-1]
        sizex, sizey = src.width, src.height
        resolution = src.res[0]

        x_values_1d = np.linspace(
            startx + 0.5 * resolution,
            startx + resolution * (sizex + 0.5),
            sizex,
            endpoint=False,
        )

        y_values_1d = np.linspace(
            starty - 0.5 * resolution,
            starty - resolution * (sizey + 0.5),
            sizey,
            endpoint=False,
        )

        x_values_2d, y_values_2d = np.meshgrid(x_values_1d, y_values_1d)
        if src.crs != "EPSG:4326":
            transformer = pyproj.Transformer.from_crs(
                src.crs, "EPSG:4326", always_xy=True
            )
            positions = np.array(
                transformer.transform(x_values_2d, y_values_2d)
            )

        distance_vector = project_coordinates_on_line(
            positions[0],
            positions[1],
            splines_dict["origin"],
            splines_dict["time_direction_vector"],
        )

        bloclen = int(distance_vector.shape[0] / 2)
        correction_1 = splines_dict["splines"](distance_vector[:bloclen, :])
        correction_2 = splines_dict["splines"](distance_vector[bloclen:, :])
        correction = np.vstack((correction_1, correction_2))

        profile = src.profile
        # User can apply correction with:
        # otbcli_BandMath -il dsm.tif correction.tif \
        # -exp "im1b1==-32768?-32768:im1b1+im2b1" -out corrected_dsm.tif
        with rio.open(
            os.path.join(out_dir, "correction.tif"), "w", **profile
        ) as dst:
            dst.write(correction, 1)


def cli():
    """
    Main cars-devibrate entrypoint (Command Line Interface)
    """
    parser = argparse.ArgumentParser(
        "cars-devibrate",
        description="Devibrate a DSM produced from stereo images",
    )

    parser.add_argument(
        "used_conf", type=str, help="CARS Used Configuration File"
    )

    parser.add_argument(
        "srtm_path",
        type=str,
        help="SRTM path",
    )

    parser.add_argument(
        "--geoid_path",
        type=str,
        help="Geoid path",
        default=GEOID_DEFAULT,
    )

    parser.add_argument(
        "--min_pts_per_time",
        type=int,
        help="minimum number of points for" "each measurement",
        default=100,
    )

    parser.add_argument(
        "--min_pts_along_time_direction",
        type=int,
        help="minimum number of points for" "time direction",
        default=100,
    )

    parser.add_argument(
        "--butterworth_filter_order",
        type=int,
        help="Order of the butterworth filter",
        default=3,
    )

    parser.add_argument(
        "--butterworth_critical_frequency",
        type=float,
        help=" The butterworth filter critical frequency",
        default=0.05,
    )

    args = parser.parse_args()
    cars_devibrate(**vars(args))


if __name__ == "__main__":
    cli()
