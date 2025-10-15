.. _installation:

============
Installation
============

--------------
Stable release
--------------

With pip
--------

CARS is available on Pypi and can be installed with pip.

It is recommended to use a virtual environnement :

.. code-block:: console

    python -m venv cars_env
    source cars_env/bin/activate
    pip install cars

This is the preferred method to install CARS, but it is required to have a Linux system.

With Docker
-----------

Alternatively, you can download the Docker image from Docker Hub

.. code-block:: console

    docker pull cnes/cars

------------
From sources
------------

The sources for CARS can be downloaded from the `Github repo <https://github.com/CNES/cars>`.

Clone the repository:

.. code-block:: console

    git clone https://github.com/CNES/cars.git

Install CARS with the Makefile :

.. code-block:: console

    make install







