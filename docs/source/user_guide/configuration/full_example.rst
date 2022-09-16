============
Full example
============

Here is a full detailed example with :ref:`orchestrator_config` and :ref:`configuration_applications` capabilities. See correspondent sections for details.

.. sourcecode:: text

    {
      "inputs": {
          "sensors" : {
              "one": {
                  "image": "img1.tif",
                  "geomodel": "img1.geom",
                  "no_data": 0
              },
              "two": {
                  "image": "img2.tif",
                  "geomodel": "img2.geom",
                  "no_data": 0

              },
              "three": {
                  "image": "img3.tif",
                  "geomodel": "img3.geom",
                  "no_data": 0
              }
          },
          "pairing": [["one", "two"],["one", "three"]],
          "initial_elevation": "srtm_dir"
        },

        "orchestrator": {
            "mode":"local_dask",
            "nb_workers": 4
        },

        "applications":{
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "sigma": 0.3
            }
        },

        "output": {
          "out_dir": "outresults"
        }
      }
