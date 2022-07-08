.. _getting_started:

===============
Getting Started
===============

.. note::

  Data samples from this tutorial can be used under `open licence <https://www.etalab.gouv.fr/licence-ouverte-open-licence>`_.

Quick Start
===========
* Install `Docker <https://docs.docker.com/get-docker/>`_
* Download `CARS Quick Start  <https://raw.githubusercontent.com/CNES/cars/master/docs/source/demo/quick_start.sh>`_

.. code-block:: console

    $ wget https://raw.githubusercontent.com/CNES/cars/master/docs/source/demo/quick_start.sh

* Run this quick_start.sh script

.. code-block:: console

    $ ./quick_start.sh

Go to the ``data_samples/outresults/`` output directory to get a :term:`DSM` and color image associated.

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

    $ wget https://raw.githubusercontent.com/CNES/cars/master/docs/source/demo/data_samples.tar.bz2
    $ wget https://raw.githubusercontent.com/CNES/cars/master/docs/source/demo/data_samples.tar.bz2.md5sum
    $ md5sum --status -c data_samples.tar.bz2.md5sum
    $ tar xvfj data_samples.tar.bz2

* Launch CARS with sensor_to_full_resolution_dsm pipeline for img1+img2 and img1+img3 pairs:

.. code-block:: console

    $ docker run -v "$(pwd)"/data_samples:/data cnes/cars /data/configfile.json

* Clean Unix rights on Docker generated data.

.. code-block:: console

    $ docker run -it -v "$(pwd)"/data_samples:/data --entrypoint /bin/bash cnes/cars -c "chown -R '$(id -u):$(id -g)' /data/"

* Show resulting output directory

.. code-block:: console

    $ ls -l data_samples/results/

.. warning::

	This first tutorial uses Docker to avoid CARS installation. To go further, follow :ref:`install` and :ref:`userguide`.


Advanced Quick Start
====================

1. :ref:`install` CARS on your system.

2. Follow now `CARS Advanced Quick Start script  <https://raw.githubusercontent.com/CNES/cars/master/docs/source/demo/quick_start_advanced.sh>`_ with the same steps than previous quick start.


.. _`QGIS`: https://www.qgis.org/
