.. _user_manual_input_data:

Input data
==========

Images and Geometric models
---------------------------

CARS supports the following official sensors raster products:

* Pléiades (PHR)
* Spot 6/7
* DigitalGlobe

More generally, all rasters for which `GDAL`_ can interpret the image geometric model through RPC coefficients may work.
For now however, CARS has been mainly tested on Pléiades products.

.. warning::
  Please check input rasters and associated **geometric model** are well read with  `OTB ReadImageInfo application <https://www.orfeo-toolbox.org/CookBook/Applications/app_ReadImageInfo.html>`_

Considering the raster images with a Dimap format (Pléiades, Spot 6/7), it is possible to directly use the XML DIMAP files. This enables to avoid a potential sub-grid division of the products, or an impeding geo-referencing of the image files (usually done for the official products), which would degrade the restitution.

An additional image can be provided to be projected on the same grid as the one of the final DSM (ortho-image).

CARS also supports the products' extracts done with the `otbcli_ExtractROI <https://www.orfeo-toolbox.org/CookBook/Applications/app_ExtractROI.html>`_ OTB application.
See :ref:`faq` for details.

Initial Input Digital Elevation Model
-------------------------------------

For now, CARS uses an initial input Digital Elevation Model which is integrated in the stereo-rectification to minimize the disparity intervals to explore.
Any geotiff file can be used.

For example, the `SRTM <https://www2.jpl.nasa.gov/srtm/>`_ data corresponding to the zone to process can be used through the `otbcli_DownloadSRTMTiles <https://www.orfeo-toolbox.org/CookBook/Applications/app_DownloadSRTMTiles.html>`_ OTB command.

Masks
------

CARS can use a mask for each image in order to ignore some image regions (for instance water mask). This mask is taken into account during the whole 3D restitution process.

The masks can be "two-states" ones: 0 values will be considered as valid data, while any other value will be considered as unvalid data and thus will be masked during the 3D restitution process.

The masks can also be multi-classes ones: they contain several values, one for each class (forest, water, cloud...). To use a multi-classes mask, a json file has to be indicated by the user in the configuration file. See the :ref:`user_manual_prepare_cli` for more details.

**Warning** : The value 255 is reserved for CARS internal use, thus no class can be represented by this value in the multi-classes masks.


.. _`GDAL`: https://gdal.org/
