============
Installation
============
.. _dependencies:

Dependencies
=============

CARS is based on the implementation of some algorithms furnished by the `Orfeo Toolbox <https://www.orfeo-toolbox.org>`_ and the `VLFeat <http://www.vlfeat.org/>`_ library.
Thus, they have to be installed in order to use CARS. See:

* `OTB installation <https://www.orfeo-toolbox.org/CookBook/Installation.html>`_
* `VLFeat installation <https://github.com/vlfeat/vlfeat>`_

After OTB installation, the following have to checked:
* all OTB environment have to be setup (otb applications working, ``OTB_APPLICATION_PATH`` set, ...)
* gdal-config have to work : If not present in your particular OTB install, copy the provided one in your OTB install path.

After Vlfeat installation, the following environment variable have to be set:
* ``VLFEAT_INCLUDE_DIR``: should be set with the path of the ``vl`` folder of the VLFeat library.
* ``VLFEAT_LIBRARY`` : should be set with the path of the ``libvl.so`` file obtained after the VLFeat library compilation.

Quick installation
==================
If dependencies are installed, CARS can be quickly installed with make command in a virtualenv environment

.. code-block:: bash

    $ git clone https://github.com/CNES/cars.git
    $ cd cars
    $ make install
    $ source venv/bin/activate
    $ source venv/bin/env_cars.sh
    $ cars -h

Configuration
=============
Cars can be configured mainly through command line : Go to :ref:`cli_usage`

A default `static configuration  <../../cars/conf/static_configuration.json>`_ is deployed with cars installation. This files enables to customize the parameters of the following algorithms:

* SIFTs computation
* alignment on the input DEM
* disparity range determination
* the points cloud filters

As well as some 3D chain parameters:

* the epipolar tiling configuration
* the grid divider factor of the rasterization step (to accelerate the neighbors searching using kd-tree)
* the output color image format
* the geometry module to use (fixed to internal `OTBGeometry`)

This file can be copied and changed with the ``CARS_STATIC_CONFIGURATION`` environment variable, which represents the full path of the changed file.

A default geoid file is installed with CARS and ``OTB_GEOID_FILE`` environment variable is automatically set.
It is possible to use another geoid by changing the location of the geoid file in ``OTB_GEOID_FILE``

Advanced Manual Installation
============================
The following steps are defined in Makefile for ``install`` command

Virtualenv
----------
First create a virtualenv and update pip

.. code-block:: bash

    $ virtualenv -p python venv/
    $ source venv/bin/activate
    $ python3 -m pip install --upgrade pip setuptools

Required python packages
------------------------

The use of CARS requires the prior installation of the some Python packages listed in the ``setup.cfg`` file and automatically installed with pip install.
But the **numpy** and **cython** package has to be installed separately otherwise some dependencies won't be correctly installed.
Also, on some installations, **fiona**, **rasterio** and **pygdal** have to be installed from source to fit local GDAL version.

See ``Makefile`` file for details.

.. code-block:: bash

    $ virtualenv -p python venv/
    $ source venv/bin/activate
    $ python3 -m pip install --upgrade cython numpy
    $ python3 -m pip install --no-binary fiona fiona
    $ python3 -m pip install --no-binary rasterio rasterio
    $ python3 -m pip install pygdal=="$(gdal-config --version).*"

It is also possible to manually install CARS correlator Pandora with different plugins and configuration: see the `Pandora documentation <https://github.com/CNES/Pandora>`_.

Environment variables
---------------------

In order to work, several environment variables impacting the dask, ITK, OTB, numba and gdal configurations are set by default in the ``env_cars.sh`` script.

For OTB CARS applications, the ``PATH``, ``PYTHONPATH``, ``LD_LIBRARY_PATH`` and ``OTB_APPLICATION_PATH`` environment variables have to be set to use the Python API as well as the OTB applications on which they depend.

CARS OTB Application  Compilation
---------------------------------

CARS OTB remote modules can be built manually (as in setup.py automatically in pip install):

.. code-block:: bash

    $ mkdir -p project_root/build
    $ cd project_root/build
    $ cmake -DOTB_BUILD_MODULE_AS_STANDALONE=ON -DCMAKE_BUILD_TYPE=Release -DVLFEAT_INCLUDE_DIR=$VLFEAT_INCLUDE_DIR ../otb_remote_module
    $ make

CARS manual installation
------------------------

Then, to install CARS:

.. code-block:: bash

    $ cd project_root
    $ pip install .

The main programs are in Python and thus can be used as they are.
