.. _quick_start:

===========
Quick start
===========

-------------
Download data
-------------

Get and extract data samples from CARS repository:

.. note::

  Data samples from this tutorial can be used under `open licence <https://www.etalab.gouv.fr/licence-ouverte-open-licence>`_.

.. code-block:: console

    wget https://raw.githubusercontent.com/CNES/cars/master/tutorials/data_gizeh.tar.bz2
    wget https://raw.githubusercontent.com/CNES/cars/master/tutorials/data_gizeh.tar.bz2.md5sum
    md5sum --status -c data_gizeh.tar.bz2.md5sum
    tar xvfj data_gizeh.tar.bz2

--------
Run CARS
--------

To launch CARS, a single configuration file is required. One is available in the data_gizeh folder: 

.. include-cars-config:: example_configs/getting_started/gizeh_configfile

With a pip installation
-----------------------

If you installed CARS with pip, the command to use is:

.. code-block:: console

    cars data_gizeh/configfile.yaml

With a Docker installation
--------------------------

If you installed CARS with Docker, the command to use is:

.. code-block:: console

    docker run -w /data -v "$(pwd)"/data_gizeh:/data cnes/cars /data/configfile.yaml

----------------
Open the results 
----------------

* Go to the ``data_gizeh/outresults/dsm`` output directory to get a :term:`DSM` (dsm.tif) and image associated (image.tif).

Open the ``dsm.tif`` DSM and ``image.tif`` image in `QGIS`_ software.

.. |dsm| image:: images/dsm.png
  :width: 100%
.. |color| image:: images/clr.png
  :width: 100%
.. |dsmcolor| image:: images/dsm_clr.png
  :width: 100%

+--------------+---------------+-------------+
|   dsm.tif    |   image.tif   | `QGIS`_ Mix |
+--------------+---------------+-------------+
| |dsm|        | |color|       | |dsmcolor|  |
+--------------+---------------+-------------+

.. _`QGIS`: https://www.qgis.org/
