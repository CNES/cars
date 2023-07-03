.. _output_postprocessing:

=======================
Output Post Processing
=======================

.. _merge_laz_files:

Merge Laz files
=====================

CARS generates multiples Laz files corresponding to processed tiles. 
Merge can be done with `laszip`_. 

To merge them:

.. code-block:: console

    $ laszip -i data\*.laz -merged -o merged.laz


.. _`laszip`: https://laszip.org/