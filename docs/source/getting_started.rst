.. _getting_started:

===============
Getting Started
===============

To get CARS started quickly, this tutorial uses Docker to simplify CARS :ref:`install`.

Data samples from this tutorial can be used under `open licence <https://www.etalab.gouv.fr/licence-ouverte-open-licence>`_.

Quick start
===========
* Install `Docker <https://docs.docker.com/get-docker/>`_
* Download `CARS Quick start  <https://raw.githubusercontent.com/CNES/cars/master/docs/source/demo/quick_start.sh>`_

.. code-block:: console

    $ wget https://raw.githubusercontent.com/CNES/cars/master/docs/source/demo/quick_start.sh

* Run this quick_start.sh script

.. code-block:: console

    $ ./quick_start.sh

Go to the ``data_samples/outcompute/`` output directory to get a :term:`DSM` from the downloaded sample data images.

Open the ``dsm.tif`` and ``color.tif`` in `QGIS <https://www.qgis.org/>`_ software.

Steps by steps
==============

* Check Docker install

.. code-block:: console

    $ docker -v

* Get CARS dockerfile image

.. code-block:: console

    $ docker pull cnes/cars

* Get and extract data samples from CARS repository:"

.. code-block:: console

    $ wget https://raw.githubusercontent.com/CNES/cars/master/docs/source/demo/data_samples.tar.bz2
    $ wget https://raw.githubusercontent.com/CNES/cars/master/docs/source/demo/data_samples.tar.bz2.md5sum
    $ md5sum --status -c data_samples.tar.bz2.md5sum
    $ tar xvfj data_samples.tar.bz2

* Run CARS :ref:`prepare_pipeline` for first pair (img1 and img2)

.. code-block:: console

    $ docker run -v "$(pwd)"/data_samples:/data cnes/cars prepare -i /data/input12.json -o /data/outprepare12

* Run CARS :ref:`prepare_pipeline` pipeline for second pair (img1 and img3)

.. code-block:: console

    $ docker run -v "$(pwd)"/data_samples:/data cnes/cars prepare -i /data/input13.json -o /data/outprepare13

* Run CARS :ref:`compute_dsm_pipeline`

.. code-block:: console

    $ docker run -v "$(pwd)"/data_samples:/data cnes/cars compute_dsm -i /data/outprepare12/content.json /data/outprepare13/content.json  -o /data/outcompute/

* Clean Unix rights on Docker generated data.

.. code-block:: console

    $ docker run -it -v "$(pwd)"/data_samples:/data --entrypoint /bin/bash cnes/cars -c "chown -R '$(id -u):$(id -g)' /data/"

* Show resulting output directory

.. code-block:: console

    $ ls -l data_samples/outcompute/

To go further, follow :ref:`install` and :ref:`user_manual`.
