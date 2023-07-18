.. include:: ../common.rst

.. _plugins:

=======
Plugins
=======

This section describes optional plugins possibilities of CARS. 

.. note::
    
    Work in progress !

.. _plugin_geometry_shareloc:

Shareloc Geometry plugin
========================

By default, the geometry functions in CARS are run through |otb|.

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

The standards parts are described in CARS :ref:`configuration`.

The particularities in the configuration file are:

* **geomodel.path**: field contain the paths to the geometric model files related to `img1` and `img2` respectively. These files have to be supported by the Shareloc library.
* **geomodel.model_type**: Depending on the nature of the geometric models indicated above, this field as to be defined as `RPC` or `GRID`. By default, "RPC"
* **initial_elevation**: Shareloc must have **a file**, typically a SRTM tile (and **not** a directory as |otb| default configuration !!)
* **geometry_plugin**: parameter configured to "SharelocGeometry" to use Shareloc plugin.


.. note::

  This library is foreseen to replace |otb| default in CARS for maintenance and installation ease.
  Be aware that geometric models must therefore be opened by shareloc directly in this case, and supported sensors may evolve.

