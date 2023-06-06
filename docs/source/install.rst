.. _install:

=======
Install
=======

VLFeat installation
===================

Besides third-parties python packages, CARS depends on `VLFeat <https://www.vlfeat.org/compiling-unix.html>`_.
If this tool is not already installed on your system, you will need to compile it as follows:

.. code-block:: console

    $ git clone https://github.com/CNES/vlfeat.git
    $ cd vlfeat && make && cd ..

then you will need to update the following environment variables for compilation:

.. code-block:: console

    $ export CFLAGS="-I$PWD/vlfeat"
    $ export LDFLAGS="-L$PWD/vlfeat/bin/glnxa64"

and the following environment variable for execution:

.. code-block:: console

    $ export LD_LIBRARY_PATH=$PWD/vlfeat/bin/glnxa64:$LD_LIBRARY_PATH


Install from PyPI (option 1)
============================

To install CARS like any other python library, you can use a virtual environment:

.. code-block:: console

    $ python -m venv venv
    $ source ./venv/bin/activate
    $ pip install --upgrade "pip<=23.0.1" "numpy>=1.17.0" cython

and install it from PyPI:

.. code-block:: console

    $ pip install cars


Install from source repository (option 2)
===============================

You can also clone CARS repository:

.. code-block:: console

    $ git clone --depth 1 https://github.com/CNES/cars.git # For latest version
    $ git clone --depth 1 --branch LAST_TAG https://github.com/CNES/cars.git # For latest stable version (replace LAST TAG by desired tag)


then, you must specify the location of the vlfeat directories:

.. code-block:: console

    $ export VLFEAT_INCLUDE_DIR=$PWD/vlfeat
    $ export VLFEAT_LIBRARY_DIR=$PWD/vlfeat/bin/glnxa64

finally, you execute makefile target:

.. code-block:: console

    $ cd cars && make install-dev-otb-free
