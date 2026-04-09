==============
DEM Generation
==============

:raw-html:`<h1>Method</h1>`

In the DEM Generation application, the goal is to create multiple Digital Elevation Models (DEMs) from previous generated DSM:
* DEM median: a median filter is applied to the filled DSM (rasterio fill nodata)
* DEM minimum: a minimum filter is applied to the filled DSM, and a negative offset is applied
* DEM maximum: a maximum filter is applied to the filled DSM, and a positive offset is applied


These DEM will be then used:
* DEM median: used as initial elevation DEM if it was not provided by user.
* DEM min and DEM max: used to estimate the disparities grids used in the Matching Applications.

:raw-html:`<h1>Limits of the method</h1>`

The method is sensible to the quality of the input DSM: if the DSM lacks data (because of occlusions, clouds, objects not visible in former lower resolution, ...).
Morevover, outliers can complete disfigure the DEMs: for DEM min and max it could result in a bad disparity estimation, but for a DEM median used in resampling, it can cause resampling aberrations.


:raw-html:`<h1>Implementation</h1>`


The low resolution DSM is filled with a rasterio fill nodata. Then Bulldozer is applied on the DSM for DEM min, and the reversed DSM for DEM max.
Then offset are applied.

Input and output geoid are applied to the DEMs if needed, through a rasterio reproject.