.. _install:

=======
Install
=======


Install from PyPI (option 1)
============================

To install CARS like any other python library, you can use a virtual environment:

.. code-block:: console

    $ python -m venv venv
    $ source ./venv/bin/activate
    $ pip install --upgrade "pip<=23.0.1" "numpy>=1.17.0" "cython<3.0.0"

and install it from PyPI:

.. code-block:: console

    $ pip install cars


Install from source repository (option 2)
=========================================

You can also clone CARS repository:

.. code-block:: console

    $ git clone --depth 1 https://github.com/CNES/cars.git # For latest version
    $ git clone --depth 1 --branch LAST_TAG https://github.com/CNES/cars.git # For latest stable version (replace LAST TAG by desired tag)

finally, you execute makefile target:

.. code-block:: console

    $ cd cars && make install-dev-otb-free
