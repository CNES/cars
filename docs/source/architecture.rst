.. _architecture:

============
Architecture
============

**CARS** is a dedicated and open source 3D tool to produce **Digital Surface Models** from satellite imaging by photogrammetry.
This Multiview stereo pipeline is intended for massive :term:`DSM` production with a robust and performant design.

It is composed of:

- A **Python 3D API**, based on xarray, enabling to realize all the computation steps leading to a :term:`DSM`.
- An **end-to-end processing** chain based on this API.

The chain can use dask (local or cluster with centralized GPFS files storage) or multiprocessing libraries to distribute the computations.

Main principles
===============

CARS is composed of well known steps in Multiview Stereo Pipeline divided in two parts to process images stereo pairs :

1. Prepare sparse pipeline : The goal is to detect data-induced failures and degraded cases in a preliminary early step.
2. Compute_dsm dense pipeline : This step is composed of the intensive parts of CARS pipeline. It takes as inputs the prepare step's outputs of several pairs and computes a unique DSM from them.

.. figure:: images/workflow.png
    :align: center
    :alt: Cars principles

    CARS Workflows

.. _prepare_pipeline:

Prepare pipeline
================

Each stereo images pair is pre-processed in an independent sparse step. From there, this first step generates refined epipolar resampling grids and an estimation of the disparity range to calibrate the heavy intensive computing part. This calibration is based on an approximate geoid (typically SRTM).
As shown in the figure above, the workflow is organized in sequential steps from input pairs (and metadata) to output data, except for the computation of sparse matches (performed tile-wise and distributed among Dask workers).

The prepare part will perform the following steps:

1. Compute the stereo-rectification grids of the input pair's images
2. Compute sift matches between the left and right images in epipolar geometry
3. Derive an optimal disparity range from the matches
4. Derive a bilinear correction model of the right image's stereo-rectification grid in order to minimize the epipolar error
5. Apply the estimated correction to the right grid
6. Export the left and corrected right grids

see :ref:`prepare_cli`

.. _compute_dsm_pipeline:

DSM compute pipeline
====================

This dense mode pipeline processes a set of pairs and computes an unique DSM from them. Each pair will be processed independently using the epipolar grids and the disparity range computed in the prepare step. A corresponding set of points clouds will then be generated. Then this set will be merged together during the rasterization step.

After prepare pipeline(s), the ``compute_dsm`` pipeline will be in charge of:

1. **resampling the images pairs in epipolar geometry** (corrected one for the right image) by using SRTM in order to reduce the disparity intervals to explore,
2. **correlating the images pairs** in epipolar geometry
3. **triangulating the sights** and get for each point of the reference image a latitude, longitude, altitude point
4. **filtering the 3D points cloud** via two consecutive filters. The first one removes the small groups of 3D points. The second filters the points which have the most scattered neighbors. Those two filters are activated by default.
5. **projecting these altitudes on a regular grid** as well as the associated color

See :ref:`compute_dsm_cli`
