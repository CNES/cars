.. _user_manual_output_data:


Output data
===========

In fine, CARS produces a geotiff file ``dsm.tif`` which contains the Digital Surface Model in the required cartographic projection and at the resolution defined by the user.

If the user provides an additional input image, an ortho-image ``clr.tif`` is also produced. The latter is stackable to the DSM (See :ref:`getting_started`).

Those two products can be visualized with `QGIS <https://www.qgis.org/fr/site/>`_ for example.

.. |dsm| image:: images/dsm.png
  :width: 100%
.. |clr| image:: images/clr.png
  :width: 100%
.. |dsmclr| image:: images/dsm_clr.png
  :width: 100%

+--------------+-------------+-------------+
|   dsm.tif    |   clr.tif   | `QGIS`_ Mix |
+--------------+-------------+-------------+
| |dsm|        | |clr|       |  |dsmclr|   |
+--------------+-------------+-------------+

CARS generates also a lot of stats described in :ref:`user_manual_compute_cli`.
