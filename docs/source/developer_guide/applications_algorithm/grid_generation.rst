===============
Grid generation
===============

:raw-html:`<h1>Method</h1>`

The grid generation application generates an epipolar grid, which is a low resolution dense grid, which can be seen as the transformation function to go from epipolar geometry to sensor geometry.

It uses initial elevation DEM and geoid, and the sensor pair associated to geometry models.

The grid is generated with a fixed epipolar step, and a defined margin for the interpolation of the grid done in the resampling application, and triangulation application.

The grid is a 2D grid for both row and column, in epipolar geometry.



:raw-html:`<h1>Implementation</h1>`

Geometry plugins construct the epipolar grid. By default, Shareloc is used : Shareloc_.


.. _Shareloc: https://shareloc.readthedocs.io/