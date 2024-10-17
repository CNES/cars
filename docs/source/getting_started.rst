.. _getting_started:

===============
Getting Started
===============

.. note::

  Data samples from this tutorial can be used under `open licence <https://www.etalab.gouv.fr/licence-ouverte-open-licence>`_.

* CARS is available on Pypi and can be installed by:

.. code-block:: console

    pip install cars

* Get and extract data samples from CARS repository:

.. code-block:: console

    wget https://raw.githubusercontent.com/CNES/cars/master/tutorials/data_gizeh.tar.bz2
    wget https://raw.githubusercontent.com/CNES/cars/master/tutorials/data_gizeh.tar.bz2.md5sum
    md5sum --status -c data_gizeh.tar.bz2.md5sum
    tar xvfj data_gizeh.tar.bz2

* Launch CARS for img1+img2 and img1+img3 pairs:

.. code-block:: console

    cars data_gizeh/configfile.json

* Configuration example for data_gizeh:

.. sourcecode:: text

    {
            "inputs": {
                "sensors" : {
                    "one": {
                        "image": "img1.tif",
                        "geomodel": "img1.geom",
                        "color": "color1.tif"
                    },
                    "two": {
                        "image": "img2.tif",
                        "geomodel": "img2.geom"
                    },
                    "three": {
                        "image": "img3.tif",
                        "geomodel": "img3.geom"
                    }
                },
                "pairing": [["one", "two"],["one", "three"]],
                "initial_elevation": "srtm_dir/N29E031_KHEOPS.tif"
            },

            "output": {
                  "out_dir": "outresults"
            }
    }

* Go to the ``data_gizeh/outresults/`` output directory to get a :term:`DSM` and color image associated.

Open the ``dsm.tif`` DSM and ``color.tif`` color image in `QGIS`_ software.

.. |dsm| image:: images/dsm.png
  :width: 100%
.. |color| image:: images/clr.png
  :width: 100%
.. |dsmcolor| image:: images/dsm_clr.png
  :width: 100%

+--------------+-----------------+---------------+
|   dsm.tif    |   color.tif     | `QGIS`_ Mix   |
+--------------+-----------------+---------------+
| |dsm|        | |color|         |  |dsmcolor|   |
+--------------+-----------------+---------------+

.. _`QGIS`: https://www.qgis.org/
