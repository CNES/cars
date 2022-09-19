.. _cli:

============
Command line
============

.. figure:: ../images/diagram_cars_overview.png
    :width: 700px
    :align: center
    :alt: CARS CLI organization

``cars`` command line is the entry point for CARS to run 3D pipelines.

It enables to run pipelines through :ref:`configuration`

.. code-block:: console

    cars -h

    usage: cars [-h] [--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--version] conf

    CARS: CNES Algorithms to Reconstruct Surface

    positional arguments:
      conf                  Inputs Configuration File

    optional arguments:
      -h, --help            show this help message and exit
      --loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                            Logger level (default: WARNING. Should be one of (DEBUG, INFO, WARNING, ERROR, CRITICAL)
      --version, -v         show program's version number and exit

Mainly, CARS cli takes only one json file as command line argument:

.. code-block:: console

    cars configfile.json
    
See :ref:`configuration` to learn how to write configuration file.



Loglevel parameter
==================

The ``loglevel`` option allows to parameter the loglevel. By default, the WARNING loglevel gives few information: only criticals, errors and warnings execution messages.

.. note::

	Use ``cars configfile.json --loglevel INFO`` to get many detailed information about each CARS steps.

.. _inputs:

Inputs
======

Images and Geometric models
---------------------------

CARS supports the following official sensors raster products:

* Pléiades (PHR)
* Spot 6/7
* DigitalGlobe

More generally, all rasters for which `GDAL`_ can interpret the image geometric model through RPC coefficients may work.
For now, however, CARS has been mainly tested on Pléiades products.

.. warning::
  Please check input rasters and associated **geometric model** are well read with  `OTB ReadImageInfo application <https://www.orfeo-toolbox.org/CookBook/Applications/app_ReadImageInfo.html>`_

Considering the raster images with a Dimap format (Pléiades, Spot 6/7), it is possible to directly use the XML DIMAP files. This enables to avoid a potential sub-grid division of the products, or an impeding geo-referencing of the image files (usually done for the official products), which would degrade the restitution.

An additional image can be provided to be projected on the same grid as the one of the final DSM (ortho-image) for radiometric superposition with the :term:`DSM`.

CARS also supports the products' extracts done with the `otbcli_ExtractROI <https://www.orfeo-toolbox.org/CookBook/Applications/app_ExtractROI.html>`_ OTB application.
See :ref:`faq` for details.

See :ref:`configuration`.

Initial Input Digital Elevation Model
-------------------------------------

For now, CARS uses an initial input Digital Elevation Model which is integrated in the stereo-rectification to minimize the disparity intervals to explore.
Any geotiff file can be used.

For example, the `SRTM <https://www2.jpl.nasa.gov/srtm/>`_ data corresponding to the zone to process can be used through the `otbcli_DownloadSRTMTiles <https://www.orfeo-toolbox.org/CookBook-7.4/Applications/app_DownloadSRTMTiles.html>`_ OTB command.

The parameter is ``initial_elevation`` as seen in :ref:`configuration`

Masks
------

CARS can use a mask for each image in order to ignore some image regions (for instance water mask). This mask is taken into account during the whole 3D restitution process.

The masks can be "two-states" ones: 0 values will be considered as valid data, while any other value will be considered as unvalid data and thus will be masked during the 3D restitution process.

The masks can also be multi-classes ones: they contain several values, one for each class (forest, water, cloud...). To use a multi-classes mask, a json file has to be indicated by the user in the configuration file. See the :ref:`configuration` for more details.

**Warning** : The value 255 is reserved for CARS internal use, thus no class can be represented by this value in the multi-classes masks.


.. _output_data:

Outputs
=======

In fine, CARS produces a geotiff file ``dsm.tif`` which contains the Digital Surface Model in the required cartographic projection and at the resolution defined by the user.

If the user provides an additional input image, an ortho-image ``clr.tif`` is also produced. The latter is stackable to the DSM (See :ref:`getting_started`).

Those two products can be visualized with `QGIS <https://www.qgis.org/fr/site/>`_ for example.

.. |dsm| image:: ../images/dsm.png
  :width: 100%
.. |clr| image:: ../images/clr.png
  :width: 100%
.. |dsmclr| image:: ../images/dsm_clr.png
  :width: 100%

+--------------+-------------+-------------+
|   dsm.tif    |   clr.tif   | `QGIS`_ Mix |
+--------------+-------------+-------------+
| |dsm|        | |clr|       |  |dsmclr|   |
+--------------+-------------+-------------+

CARS generates also a lot of stats.


.. _`GDAL`: https://gdal.org/


Simple example
==============

A simple json file with only required configuration:

.. sourcecode:: text

    {
      "inputs": {
          "sensors" : {
              "one": {
                  "image": "img1.tif",
                  "geomodel": "img1.geom"
              },
              "two": {
                  "image": "img2.tif",
                  "geomodel": "img2.geom"

              }
          },
          "pairing": [["one", "two"]]
      },
      "output": {
          "out_dir": "outresults"
        }
    }

Launch CARS with configuration file

.. code-block:: console

   cars configfile.json
