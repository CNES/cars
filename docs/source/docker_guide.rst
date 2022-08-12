.. _docker_guide:

============
Docker guide
============

This page only gives element to use and build CARS dockers.

Please go to `Docker`_ documentation for more details on Docker technology.

* First, check Docker install, needed for this page.

.. code-block:: console

    $ docker -v

Usage
=====

CARS can  be used more easily through a docker to avoid the complete installation (with OTB and VLFeat).

* Install `Docker`_

* Get public CARS dockerfile images

Main CARS docker image
----------------------

* Get official CARS docker image

.. code-block:: console

    $ docker pull cnes/cars
    $ docker pull cnes/cars:0.5.0 # for a particular cars docker version

    
* Run CARS CLI

.. code-block:: console

    $ docker run cnes/cars # for CARS command line directly by default "cars -h"
      

CARS Jupyter notebook docker image
----------------------------------

* Get official CARS Jupyter docker image

.. code-block:: console

    $ docker pull cnes/cars-jupyter
    $ docker pull cnes/cars-jupyter:0.5.0 # for a particular version

* Run CARS CLI

.. code-block:: console

    $ docker run -p 8888:8888 cnes/cars-jupyter
    
This runs a jupyter notebook directly to *https://localhost:8888/*

Follow output indications.


CARS tutorial slideshow
-----------------------

* Get official CARS tutorial docker image

.. code-block:: console

    $ docker pull cnes/cars-tutorial
    $ docker pull cnes/cars-tutorial:0.5.0 # for a particular version

* Run CARS CLI

.. code-block:: console

    $ docker run -p 8000:8000 cnes/cars-tutorial

You can go now to *https://localhost:8000/*


Build
=====

Instead of pulling dockerhub official CARS images, here is the way to build images:

* Clone CARS repository from GitHub :

.. code-block:: console

    $ git clone https://github.com/CNES/cars.git
    $ cd cars

* Check and build CARS docker image

.. code-block:: console

    $ make docker
    
  The command:
  
  * checks Dockerfiles with hadolint
  * builds locally CARS main docker image: cnes/cars
  * builds locally CARS jupyter wrapper docker image: cnes/cars-jupyter
  * builds locally CARS tutorial wrapper to jupyter docker image: cnes/cars-tutorial

See section *docker* in `Makefile  <https://raw.githubusercontent.com/CNES/cars/master/Makefile>`_ for more details.

.. _`Docker`: https://docs.docker.com/