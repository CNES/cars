.. _install:

=======
Install
=======
.. _dependencies:

Dependencies
=============

Besides thirparties python packages, CARS depends on `OTB <https://www.orfeo-toolbox.org/CookBook/Installation.html>`_ and `VLFeat <https://www.vlfeat.org/compiling-unix.html>`_

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

* Clone CARS source code (choose version)

.. code-block:: console

    $ git clone --depth 1 https://github.com/CNES/cars.git # For latest version
    $ git clone --depth 1 --branch LAST_TAG https://github.com/CNES/cars.git # For latest stable version

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
This chapter further explains the content and behavior of ``make install`` (see the `Makefile <https://raw.githubusercontent.com/CNES/cars/master/Makefile>`_).

Virtualenv
----------
First create a virtualenv and upgrade main pip packages.

.. code-block:: console

    $ virtualenv -p python venv/
    $ source venv/bin/activate
    $ python3 -m pip install --upgrade pip setuptools

Required python packages
------------------------

CARS python package requires some python packages:

* **numpy**, **cython**: They have to be installed at first, otherwise some dependencies won't be correctly installed.
* **fiona**, **rasterio**, **pygdal**: On some systems, they have to be installed from sources to match local GDAL version.

Here are the corresponding commands to install these requirements:

.. code-block:: console

    $ virtualenv -p python venv/
    $ source venv/bin/activate
    $ python3 -m pip install --upgrade cython numpy
    $ python3 -m pip install --no-binary fiona fiona
    $ python3 -m pip install --no-binary rasterio rasterio
    $ python3 -m pip install pygdal=="$(gdal-config --version).*"

To manually install CARS core dense matching correlator, see the `Pandora documentation <https://github.com/CNES/Pandora>`_.

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

You can use ``pip install .[docs]`` and/or ``pip install .[dev]`` to install specific dependencies that are not required to run CARS.

Follow :ref:`user_guide` to run and configure CARS.

CARS OTB Application Compilation
--------------------------------
This compilation is automatically done through CARS pip install.

Nonetheless, CARS internal OTB remote modules can be built manually if needed:

.. code-block:: console

    $ mkdir -p project_root/build
    $ cd project_root/build
    $ cmake -DOTB_BUILD_MODULE_AS_STANDALONE=ON -DCMAKE_BUILD_TYPE=Release -DVLFEAT_INCLUDE_DIR=$VLFEAT_INCLUDE_DIR ../otb_remote_module
    $ make
