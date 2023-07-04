.. _output_postprocessing:

=======================
Output Post Processing
=======================

.. _merge_laz_files:

Merge Laz files
=====================

CARS generates several laz files corresponding to the tiles processed.
Merge can be done with `laszip`_. 

To merge them:

.. code-block:: console

    $ laszip -i data\*.laz -merged -o merged.laz


.. _`laszip`: https://laszip.org/