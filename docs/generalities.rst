Main principles
===============

CARS is composed of well known steps in Multiview Stereo Pipeline divided in two parts to process images stereo pairs :

1. Prepare sparse step : The goal is to detect data-induced failures and degraded cases in a preliminary early step.
2. Compute_dsm dense step : This step is composed of the intensive parts of CARS pipeline. It takes as inputs the prepare step's outputs of several pairs and computes a unique DSM from them.

.. figure:: images/workflow.png
    :align: center
    :alt: Cars principles

    CARS Workflows

Preparation step
----------------

Each stereo images pair is pre-processed in an independent sparse step. From there, this first step generates refined epipolar resampling grids and an estimation of the disparity range to calibrate the heavy intensive computing part. This calibration is based on an approximate geoid (typically SRTM).
As shown in the figure above, the workflow is organized in sequential steps from input pairs (and metadata) to output data, except for the computation of sparse matches (performed tile-wise and distributed among Dask workers). 

DSM computation step
--------------------

This dense step processes a set of pairs and computes an unique DSM from them. Each pair will be processed independently using the epipolar grids and the disparity range computed in the prepare step. A corresponding set of points clouds will then be generated. Then this set will be merged together during the rasterization step.

Input data
==========

Images
------
CARS supports the following official products: Pléiades, Spot 6/7, DigitalGlobe sensors and, more generally, all images for which gdal can interpret the RPC coefficients.

Considering the images with a Dimap format (Pléiades, Spot 6/7), it is possible to directly use the XML DIMAP files. This enables to avoid a potential sub-grid division of the products, or an impeding geo-referencing of the image files (usually done for the official products), which would degrade the restitution.

An additional image can be provided to be projected on the same grid as the one of the final DSM (ortho-image).

CARS also supports the products' extracts done with the ``otbcli_ExtractROI`` OTB application.

To prepare such a data set, retrieve the coordinates of the desired extract on the first image (lets call it ``img1.jp2``), under the form ``startx``, ``starty``, ``sizex``, ``sizey`` (in pixels).

Perform the extraction of the first image with:

.. code-block:: bash

    $ otbcli_ExtractROI -in img1.jp2 -out img1_xt.tif uint16 -startx startx -starty starty -sizex sizex -sizey sizey

To extract the same zone on the second image (for example ``img2.jp2``), the ``-mode fit`` application option has to be used:

.. code-block:: bash

    $ otbcli_ExtractROI -in img2.jp2 -out img2_xt.tif uint16 -mode fit -mode.fit.im img1_xt.tif

The application will automatically look for the zone corresponding to ``img1_xt.tif`` within ``img2.jp2``.

It is possible to use the ``-elev.dem srtm/`` option to use the DEM during this search in order to be more accurate.

It is to be noted that the ``-mode.fit.vec`` option also exists. It accepts a vector file (for example a shapefile or a kml) which enables the image extraction from a footprint.

Initial Digital Elevation Model
-------------------------------

For now, CARS uses an initial Digital Elevation Model which is integrated in the stereo-rectification to minimize the disparity intervals to explore. Any geotiff file can be used. If needed, the ``otbcli_DownloadSRTMTiles`` OTB command enables to download the SRTM data corresponding to the zone to process.

Masks
-----

CARS can use a mask for each image in order to ignore some image regions (for instance water mask). This mask is taken into account during the whole 3D restitution process.

The masks can be "two-states" ones: 0 values will be considered as valid data, while any other value will be considered as unvalid data and thus will be masked during the 3D restitution process.

The masks can also be multi-classes ones: they contain several values, one for each class (forest, water, cloud...). To use a multi-classes mask, a json file has to be indicated by the user in the configuration file. See the [cli_usage](./cli_usage.rst) for more details.

**Warning** : The value 255 is reserved for CARS internal use, thus no class can be represented by this value in the multi-classes masks.


Output data
===========

In fine, CARS produces a geotiff file which contains the Digital Surface Model in the required cartographic projection and at the resolution defined by the user.

If the user provided an additional image, an ortho-image is also produced. The latter is stackable to the DSM.

Those two products can be visualized with `QGIS <https://www.qgis.org/fr/site/>`_ for example.

Considering bulky files, it is recommended to generate an overview file with `GDAL <https://gdal.org/>`_ before opening it with QGIS:

.. code-block:: bash

    $ gdaladdo -ro -r average dsm.tif 2 4 8 16 32 64
