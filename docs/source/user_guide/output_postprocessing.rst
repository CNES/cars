.. _output_postprocessing:

=======================
Output Post Processing
=======================

.. _merge_laz_files:

Merge Laz files
=====================

CARS generates multiples Laz files corresponding to processed tiles. To merge them:

.. code-block:: console

    $ laszip -i data\*.las -merged -o merged.laz