.. _user_manual_advanced_configuration:

======================
Advanced configuration
======================


Static Configuration
====================

A default static configuration `static_configuration.json <https://raw.githubusercontent.com/CNES/cars/master/cars/conf/static_configuration.json>`_ is deployed with CARS :ref:`install`.

This file enables to customize the parameters of the following algorithms:

* SIFTs computation
* alignment on the input DEM
* disparity range determination
* the points cloud filters
* the epipolar tiling configuration
* the grid divider factor of the rasterization step (to accelerate the neighbors searching using kd-tree)
* the output color image format
* the geometry loader to use (fixed to internal `OTBGeometry`)
* the geoid to use in CARS (optional). The user can choose to use CARS internal geoid file by setting `use_cars_geoid` to true or provides his own geoid via the `geoid_path` key. To not use any geoid file, the user have to set `use_cars_geoid` to false and fix the `geoid_path` to `None` or not set it at all.

This file can be copied and changed with the ``CARS_STATIC_CONFIGURATION`` environment variable, which represents the full path of the changed file.

Geoid Configuration
===================

A default geoid file is installed with CARS and ``OTB_GEOID_FILE`` environment variable is automatically set.
It is possible to use another geoid by changing the location of the geoid file in ``OTB_GEOID_FILE``
