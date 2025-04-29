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
this module contains tools for the dem generation
"""

import contextlib
import logging
import os

# Third party imports
import xdem


def fit_initial_elevation_on_dem_median(
    dem_to_fit_path: str, dem_ref_path: str, dem_out_path: str
):
    """
    Coregistrates the two DEMs given then saves the result.
    The initial elevation will be cropped to reduce computation costs.
    Returns the transformation applied.

    :param dem_to_fit_path: Path to the dem to be fitted
    :type dem_to_fit_path: str
    :param dem_ref_path: Path to the dem to fit onto
    :type dem_ref_path: str
    :param dem_out_path: Path to save the resulting dem into
    :type dem_out_path: str

    :return: coregistration transformation applied
    :rtype: dict
    """
    # suppress all outputs of xdem
    with open(os.devnull, "w", encoding="utf8") as devnull:
        with (
            contextlib.redirect_stdout(devnull),
            contextlib.redirect_stderr(devnull),
        ):

            # load DEMs
            dem_to_fit = xdem.DEM(dem_to_fit_path)
            dem_ref = xdem.DEM(dem_ref_path)  # 0s are nodata in dem_ref

            # get the crs needed to reproject the data
            crs_out = dem_ref.crs
            crs_metric = dem_ref.get_metric_crs()

            # Crop dem_to_fit with dem_ref to reduce
            # computation costs. This is fine since dem_ref has big margins
            # and we want to fix small shifts.
            bbox = dem_ref.bounds
            dem_to_fit = dem_to_fit.crop(bbox).reproject(crs=crs_metric)
            # Reproject dem_ref to dem_to_fit resolution to reduce
            # computation costs
            dem_ref = dem_ref.reproject(dem_to_fit)

            coreg_pipeline = xdem.coreg.NuthKaab()

            try:
                # fit dem_to_fit onto dem_ref, crop it, then reproject it
                # set a random state to always get the same results
                fit_dem = (
                    coreg_pipeline.fit_and_apply(
                        dem_ref, dem_to_fit, random_state=0
                    )
                    .crop(dem_ref)
                    .reproject(crs=crs_out)
                )
                # save the results
                fit_dem.save(dem_out_path)
                coreg_offsets = coreg_pipeline.meta["outputs"]["affine"]
            except (ValueError, AssertionError):
                logging.warning(
                    "xDEM coregistration failed. This can happen when sensor "
                    "images are too small. No shift will be applied on DEM"
                )
                coreg_offsets = None

    return coreg_offsets
