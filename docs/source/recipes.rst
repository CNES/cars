.. _recipes:

=======
Recipes
=======

Input image preparation
=======================

.. _make_input_roi_images:

Make input ROI images
---------------------

CARS supports the :term:`ROI` products extracts done with the `otbcli_ExtractROI <https://www.orfeo-toolbox.org/CookBook/Applications/app_ExtractROI.html>`_ OTB application (raster + geometric model).

* First, retrieve the coordinates of the desired extract on the first image (lets call it ``img1.jp2``), under the form ``startx``, ``starty``, ``sizex``, ``sizey`` (in pixels).

* Perform the extraction of the first image with:

.. code-block:: console

    otbcli_ExtractROI -in img1.jp2 -out img1_xt.tif uint16 -startx startx -starty starty -sizex sizex -sizey sizey

* Extract the same zone on the second image (for example ``img2.jp2``), the ``-mode fit`` application option has to be used:

.. code-block:: console

    otbcli_ExtractROI -in img2.jp2 -out img2_xt.tif uint16 -mode fit -mode.fit.im img1_xt.tif

The application will automatically look for the zone corresponding to ``img1_xt.tif`` within ``img2.jp2``.

It is possible to use the ``-elev.dem srtm/`` option to use the DEM during this search in order to be more accurate.

It is to be noted that the ``-mode.fit.vec`` option also exists. It accepts a vector file (for example a shapefile or a kml) which enables the image extraction from a footprint.

.. _make_a_simple_pan_sharpening:

Make a simple pan sharpening
----------------------------

In the case of Pleiades sensors, the XS color isn't superimposable to the Panchromatic image.

It can be recommended to apply a P+XS pansharpening with `OTB`_.

.. code-block:: console

    otbcli_BundleToPerfectSensor -inp image.tif -inxs color.tif -out color_pxs.tif

.. _`OTB`: https://www.orfeo-toolbox.org/CookBook-8.0/C++/UserGuide.html#image-data-representation

.. _make_a_water_mask:

Make a water mask
-----------------

To produce a water mask from R,G,B,NIR images, it can be recommended to compute a Normalized Difference Water Index (NDWI) and threshold the output to a low value.

The low NDWI values can be considered as water area.

.. code-block:: console

    gdal_calc.py -G input.tif --G_band=2 -N input.tif --N_band=4 --outfile=mask.tif --calc="((1.0*G-1.0*N)/(1.0*G+1.0*N))>0.3" --NoDataValue=0

.. _`GDAL`: https://gdal.org/

See next section to apply a gdal translate to convert the mask with 1bit image struture.

.. _convert_image_to_binary_image:

Convert image to binary image
-----------------------------

To translate single image or multiband image with several nbits per band to 1bit per band, it can be recommended to use gdal_translate as follows:

.. code-block:: console

    gdal_translate -ot Byte -co NBITS=1 mask.tif mask_1nbit.tif

.. _`GDAL`: https://gdal.org/

.. _add_band_description_in_image:

Add band name / description in tiff files metadata
--------------------------------------------------

To add a band name /description in tiff files, for classification or color files in order to be used:


.. code-block:: python

    data_in = gdal.Open(infile, gdal.GA_Update)
    band_in = data_in.GetRasterBand(inband)
    band_in.SetDescription(band_description)
    data_in = None


Output Post Processing
======================

.. _merge_laz_files:

Merge Laz files
---------------

CARS generates several laz files corresponding to the tiles processed.
Merge can be done with `laszip`_. 

To merge them:

.. code-block:: console

    laszip -i data\*.laz -merged -o merged.laz


.. _`laszip`: https://laszip.org/
