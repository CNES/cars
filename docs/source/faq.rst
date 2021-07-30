.. _faq:

==========================
Frequently Asked Questions
==========================


Input data
==========

How to create CARS compatible :term:`ROI` input data with OTB ?
---------------------------------------------------------------

CARS supports the :term:`ROI` products extracts done with the `otbcli_ExtractROI <https://www.orfeo-toolbox.org/CookBook/Applications/app_ExtractROI.html>`_ OTB application (raster + geometric model).

* First, retrieve the coordinates of the desired extract on the first image (lets call it ``img1.jp2``), under the form ``startx``, ``starty``, ``sizex``, ``sizey`` (in pixels).

* Perform the extraction of the first image with:

.. code-block:: console

    $ otbcli_ExtractROI -in img1.jp2 -out img1_xt.tif uint16 -startx startx -starty starty -sizex sizex -sizey sizey

* Extract the same zone on the second image (for example ``img2.jp2``), the ``-mode fit`` application option has to be used:

.. code-block:: console

    $ otbcli_ExtractROI -in img2.jp2 -out img2_xt.tif uint16 -mode fit -mode.fit.im img1_xt.tif

The application will automatically look for the zone corresponding to ``img1_xt.tif`` within ``img2.jp2``.

It is possible to use the ``-elev.dem srtm/`` option to use the DEM during this search in order to be more accurate.

It is to be noted that the ``-mode.fit.vec`` option also exists. It accepts a vector file (for example a shapefile or a kml) which enables the image extraction from a footprint.
