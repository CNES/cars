.. _install:

=======
Install
=======
.. _dependencies:

Dependencies
=============

CARS depends on `OTB <https://www.orfeo-toolbox.org/CookBook/Installation.html>`_ and `VLFeat <https://www.vlfeat.org/compiling-unix.html>`_

* Check OTB install:

  * OTB environment has to be setup:

    * OTB applications are working. Example: ``otbcli_ReadImageInfo -in some_image.tif``
    * ``OTB_APPLICATION_PATH`` is set.
    * GDAL must work (gdal-config --version must be available)

* Check Vlfeat install with following global environment variables:

  * ``VLFEAT_INCLUDE_DIR``: should be set with the path of the ``vl`` folder of the VLFeat library.
  * ``VLFEAT_LIBRARY``: should be set with the path of the ``libvl.so`` file obtained after the VLFeat library compilation.

See `CARS Dockerfile <https://raw.githubusercontent.com/CNES/cars/master/Dockerfile>`_ example for detailed steps.

Quick install
=============

* Clone CARS Source code (choose version)

.. code-block:: console

    $ git clone --depth 1 https://github.com/CNES/cars.git # For latest version
    $ git clone --depth 1 --branch LAST_TAG https://github.com/CNES/cars.git # For last stable version

* Install CARS

.. code-block:: console

    $ cd cars
    $ make install  # Cars is installed in `venv` directory

* Run CARS in virtualenv

.. code-block:: console

    $ source venv/bin/activate
    $ source venv/bin/env_cars.sh
    $ cars -h

Advanced Install
================
The following steps are defined in `Makefile <https://raw.githubusercontent.com/CNES/cars/master/Makefile>`_  ``install`` command.

Virtualenv
----------
First create a virtualenv and upgrade main pip packages.

.. code-block:: console

    $ virtualenv -p python venv/
    $ source venv/bin/activate
    $ python3 -m pip install --upgrade pip setuptools

Required python packages
------------------------

CARS python package requires some python packages to be installed before:

* **numpy**, **cython**: They have to be installed separately otherwise some dependencies won't be correctly installed.
* **fiona**, **rasterio**, **pygdal**: On some systems, they have to be installed from source to fit local GDAL version.

Here are the correspondent commands to install these prior dependencies:

.. code-block:: console

    $ virtualenv -p python venv/
    $ source venv/bin/activate
    $ python3 -m pip install --upgrade cython numpy
    $ python3 -m pip install --no-binary fiona fiona
    $ python3 -m pip install --no-binary rasterio rasterio
    $ python3 -m pip install pygdal=="$(gdal-config --version).*"

To manually install CARS core correlator, see the `Pandora documentation <https://github.com/CNES/Pandora>`_.

Environment variables
---------------------

The script `env_cars.sh <https://raw.githubusercontent.com/CNES/cars/master/env_cars.sh>`_ sets several environment variables impacting dask, ITK, OTB, numba and GDAL configurations.

For CARS internal OTB remote modules, the ``PATH``, ``PYTHONPATH``, ``LD_LIBRARY_PATH`` and ``OTB_APPLICATION_PATH`` environment variables have also to be set.

CARS manual install
-------------------

Then, to install CARS:

.. code-block:: console

    $ cd project_root
    $ pip install .

Follow :ref:`userguide` to run and configure CARS.

CARS OTB Application Compilation
--------------------------------
This compilation is automatically done through CARS pip install.

Nonetheless, CARS internal OTB remote modules can be built manually if needed:

.. code-block:: console

    $ mkdir -p project_root/build
    $ cd project_root/build
    $ cmake -DOTB_BUILD_MODULE_AS_STANDALONE=ON -DCMAKE_BUILD_TYPE=Release -DVLFEAT_INCLUDE_DIR=$VLFEAT_INCLUDE_DIR ../otb_remote_module
    $ make
