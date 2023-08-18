.. include:: ../common.rst

.. _plugins:

=======
Plugins
=======

This section describes optional plugins possibilities of CARS. 

.. note::
    
    Work in progress !

.. tabs::

    .. tab:: OTB Geometry plugin

        By default, the geometry functions in CARS are run through |otb|.

        To use OTB geometry library, CARS input configuration should be defined as :

        .. code-block:: json

            {
              "inputs": {
                "sensors": {
                  "one": {
                    "image": "img1.tif",
                    "geomodel": {
                      "path": "img1.geom"
                    },
                  },
                  "two": {
                    "image": "img2.tif",
                    "geomodel": {
                      "path": "img2.geom"
                    },
                  }
                },
                "pairing": [["one", "two"]],
                "initial_elevation": "path/to/srtm_file"
              },
              "geometry_plugin": "OTBGeometry",
              "output": {
                "out_dir": "outresults"
              }
            }

        The standards parts are described in CARS :ref:`configuration`.

        The particularities in the configuration file are:

        * **geomodel.path**: Field contains the paths to the geometric model files related to `img1` and `img2` respectively.
        * **initial_elevation**: Field contains the path to the **folder** in which are located the SRTM tiles covering the production.
        * **geometry_plugin**: Parameter configured to "OTBGeometry" to use OTB library.

        Parameter can also be defined as a string *path* instead of a dictionary in the configuration. In this case, geomodel parameter will
        be changed to a dictionary before launching the pipeline. The dictionary will be :

        .. code-block:: json

            {
              "path": "img1.geom"
            }

    .. tab:: Shareloc Geometry plugin

        Another geometry library called `Shareloc`_ is installed with CARS and can be configured to be used as another option.

        To use Shareloc library, CARS input configuration should be defined as :

        .. code-block:: json

            {
              "inputs": {
                "sensors": {
                  "one": {
                    "image": "img1.tif",
                    "geomodel": {
                      "path": "img1.geom",
                      "model_type": "RPC"
                    },
                  },
                  "two": {
                    "image": "img2.tif",
                    "geomodel": {
                      "path": "img2.geom",
                      "model_type": "RPC"
                    },
                  }
                },
                "pairing": [["one", "two"]],
                "initial_elevation": "path/to/srtm_file"
              },
              "geometry_plugin": "SharelocGeometry",
              "output": {
                "out_dir": "outresults"
              }
            }

        The particularities in the configuration file are:

        * **geomodel.model_type**: Depending on the nature of the geometric models indicated above, this field as to be defined as `RPC` or `GRID`. By default, "RPC".
        * **initial_elevation**: Field contains the path to the **file** corresponding the srtm tiles covering the production (and **not** a directory as OTB default configuration !!)
        * **geometry_plugin**: Parameter configured to "SharelocGeometry" to use Shareloc plugin.

        Parameter can also be defined as a string *path* instead of a dictionary in the configuration. In this case, geomodel parameter will
        be changed to a dictionary before launching the pipeline. The dictionary will be :

        .. code-block:: json

            {
              "path": "img1.geom",
              "model_type": "RPC"
            }


.. note::

  This library is foreseen to replace |otb| default in CARS for maintenance and installation ease.
  Be aware that geometric models must therefore be opened by Shareloc directly in this case, and supported sensors may evolve.

