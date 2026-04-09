=============
Rasterization
=============

:raw-html:`<h1>Method</h1>`

The previously generated point clouds are rasterized in this Rasterization application.

The application defines the regular grid dimensions, used for the rasterization. It defines the metadata of output raster (dimensions, geotransform, projection).

Then for each depth map tile, the point cloud is rasterized with the following method:

  * For each pixel of the output raster, we identify the neighboring cells  with a given radius, and get all the points contained in these cells.
  * We compute the euclidian distance between the pixel center and the points.
  * We apply a gaussian weighting function to the distance, controled with sigma parameter: the greater sigma, the more far points will be taken into account.
  * An auxiliary raster stores the used weight for each pixel, used during the merging of rasterized tiles.
  * Other auxiliary rasters are generated such as statistics rasters (number of points, mean distance, etc) for quality control.

:raw-html:`<h1>Limits of the method</h1>`


:raw-html:`<h1>Implementation</h1>`

The rasterization is done using cars-rasterize library, in the wrapper.


The particularity of this application is that it is performed in 2 steps:
  * by depth map tile, alone, in parallel.
  * by merging the rasterized tile with the previous merged rasterized tiles already dumped in tif file.

For the second step, we used the merging function passed to the orchestrator, to be applied when it receives the rasterized tile.
To merge the rasterized tile with the previous merged rasterized tile, we use the auxiliary raster of weights: for each pixel, we compute the new value as a weighted average of the previous merged value and the new rasterized tile value, with the weights of both rasters.


