=============
Triangulation
=============

:raw-html:`<h1>Method</h1>`

In the Triangulation application, we are given:

  * A CarsDataset containing a Disparity Map between a left and right epipolar image.
  * The Epipolar Geometry associated to the left and right epipolar images.
  * A geometry plugin used to generate the epipolar geometry used for resampling and sparse/dense matching.
  * The epipolar grids associated to the left and right epipolar images.

The Triangulation is performed with Shareloc by default: `https://github.com/CNES/shareloc <https://github.com/CNES/shareloc>`_ .

We triangulate each tile of the Disparity Map CarsDataset, to generate a 3D point cloud tile.
For each tile, we use the geometry plugin to perform the triangulation:

   1. Transform epipolar coordinates (line, column) to image coordinates (x, y) for left and right images, using the epipolar grids.
   2. Perform line of sight intersection for each match in sensor geometry.
   3. Generate a point cloud tile containing 3D points (X, Y, Z) in the chosen coordinate system.


Each disparity map is triangulated, including the disparity inf and sup used to compute performance maps.



:raw-html:`<h1>Limits of the method</h1>`


:raw-html:`<h1>Implementation</h1>`

The Triangulation application is implemented in the file ``cars/applications/triangulation/line_of_sight_intersection.py``. The applications generated a CarsDataset of type "array", containing all the tiles of the 3D point cloud.

Every tile is computed in parallel, using the Orchestrator framework. The wrapper function used for each tile, generates a xarray Dataset containing the 3D point cloud tile, as x, y, and z layers, stored in epipolar geometry.
We call this storage a "depth map".

This depth map can be transformed in a point cloud, and saved as a laz file, or csv file.

The Laz files are saved, with each tile in a separate file, in parallel. This is the opposite method of the writting of tif files, done in the main process.