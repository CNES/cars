.. _getting_started:

===============
Getting Started
===============

.. note::

  Data samples from this tutorial can be used under `open licence <https://www.etalab.gouv.fr/licence-ouverte-open-licence>`_.

Quick Start
===========
* Install `Docker <https://docs.docker.com/get-docker/>`_
* Download `CARS Quick Start  <https://raw.githubusercontent.com/CNES/cars/master/tutorials/quick_start.sh>`_

.. code-block:: console

    $ wget https://raw.githubusercontent.com/CNES/cars/master/tutorials/quick_start.sh

* Run this quick_start.sh script

.. code-block:: console

    $ ./quick_start.sh

Go to the ``data_gizeh/outresults/`` output directory to get a :term:`DSM` and color image associated.

Open the ``dsm.tif`` DSM and ``clr.tif`` color image in `QGIS`_ software.

.. |dsm| image:: images/dsm.png
  :width: 100%
.. |clr| image:: images/clr.png
  :width: 100%
.. |dsmclr| image:: images/dsm_clr.png
  :width: 100%

+--------------+-------------+-------------+
|   dsm.tif    |   clr.tif   | `QGIS`_ Mix |
+--------------+-------------+-------------+
| |dsm|        | |clr|       |  |dsmclr|   |
+--------------+-------------+-------------+

Steps by steps
==============

* Check Docker install

.. code-block:: console

    $ docker -v

* Get CARS dockerfile image

.. code-block:: console

    $ docker pull cnes/cars

* Get and extract data samples from CARS repository:

.. code-block:: console

    $ wget https://raw.githubusercontent.com/CNES/cars/master/tutorials/data_gizeh.tar.bz2
    $ wget https://raw.githubusercontent.com/CNES/cars/master/tutorials/data_gizeh.tar.bz2.md5sum
    $ md5sum --status -c data_gizeh.tar.bz2.md5sum
    $ tar xvfj data_gizeh.tar.bz2

* Launch CARS for img1+img2 and img1+img3 pairs:

.. code-block:: console

    $ docker run -w /data -v "$(pwd)"/data_gizeh:/data cnes/cars /data/configfile.json

* Configuration example for quick_start data_gizeh:

.. sourcecode:: text

    {
            "inputs": {
                "sensors" : {
                    "one": {
                        "image": "img1.tif",
                        "geomodel": "img1.geom",
                        "color": "color1.tif",
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

            "output": {
                  "out_dir": "outresults"
            }
    }


* Clean Unix rights on Docker generated data.

.. code-block:: console

    $ docker run -it -v "$(pwd)"/data_gizeh:/data --entrypoint /bin/bash cnes/cars -c "chown -R '$(id -u):$(id -g)' /data/"

* Show resulting output directory

.. code-block:: console

    $ ls -l data_gizeh/outresults/

.. warning::

	This first tutorial uses Docker to avoid CARS installation. To go further, follow :ref:`install` and :ref:`user_guide`.


Advanced Quick Start
====================

1. :ref:`install` CARS on your system directly.

2. Follow now `CARS Advanced Quick Start script  <https://raw.githubusercontent.com/CNES/cars/master/tutorials/quick_start_advanced.sh>`_ with the same steps than previous quick start.

The main steps are:

* Download inputs and extract them

.. code-block:: console

    $ wget https://raw.githubusercontent.com/CNES/cars/master/tutorials/data_gizeh.tar.bz2
    $ wget https://raw.githubusercontent.com/CNES/cars/master/tutorials/data_gizeh.tar.bz2.md5sum
    $ md5sum --status -c data_gizeh.tar.bz2.md5sum
    $ tar xvfj data_gizeh.tar.bz2

* Launch CARS

.. code-block:: console

    $ cars data_gizeh/configfile.json

Tutorials and notebooks
=======================

CARS provides a full python API which can be used to compute :term:`DSM` step by step and gives access to intermediate data.

If you want to learn how to use it, go to the available `notebooks tutorials <https://github.com/CNES/cars/tree/master/tutorials>`_

To ease the use, follow :ref:`docker_guide` guide to be able to get the tutorials easily.


.. _`QGIS`: https://www.qgis.org/



