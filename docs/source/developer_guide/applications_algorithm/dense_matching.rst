==============
Dense Matching
==============

:raw-html:`<h1>Method</h1>`

In the Dense Matching Application, we can given a pair of left and right epipolar images.
By construction from resampling, these images are stored in tiles inside a CarsDataset. Each tile contains:

  * the tile region of interest: area where Matches will be found
  * the overlap area with neighboring tiles: area used to find Matches close to tile borders.

In this application, we consider that we have a perfect epipolar geometry. It allows us to search for matches only along the epipolar lines.


The correspondence search is performed using Pandora. `https://github.com/CNES/Pandora <https://github.com/CNES/Pandora>`_ .
This library implements several dense matching algorithms. The default one used in CARS is the Semi-Global Matching (SGM) algorithm, with the Census Transform cost volume.

The following steps are performed for each tile in Pandora:

  1.  Matching cost volume computation: for each pixel in the left image, and for each disparity in the disparity range, a cost is computed using for instance the Census Transform, znnc, of Mccnn.
  2.  Cost volume aggregation.
  3.  Cost volume optimization using SGM.
  4.  Disparity map computation.
  5.  Refinement of the disparity map using subpixel interpolation.
  6.  Filtering of the disparity map.
  7.  Generation of confidence map (used in CARS) : Ambiguity, Risk, intervals. These maps are used in CARS to generate the performance maps.
  8.  Cross validation. if a left-right image pair is given, a left-right consistency check is performed to filter out inconsistent disparities.


The margin used is defined according to;
    * the Pandora methods: SGM requires a certain margin to perform correctly the "semi-global" optimization.
    * the disparity range to explore : each left pixel should have all its possible matches in the right image inside the tile (including the margin).


**Performance maps**

Several products can be generated in addition to the disparity map:

  * Ambiguity map: in [0, 1] : gives for each pixel the confidence in the selected pic. To put it very simply, the more pics there are, the less confident we are.
  * Risk map: gives an estimation of the uncertainty on the disparity value for each pixel.
  * Interval map: gives for each pixel, the interval of disparity values that are considered good matches (based on a cost threshold).

Then the performance map is produced by CARS, by triangulating the interval map (disparity inf and sup), and combining it with the ambiguity map.


:raw-html:`<h1>Limits of the method</h1>`

The main limit is the memory consumption of Pandora for large disparity ranges, due to the cost volume size.
An optimal tile size is computed according to the disparity range, and the available memory per worker. However, we cannot go bellow a certain tile size.  In this case, the disparity range is cropped.

Moreover, Pandora must be given rectified images, withe the exact same size. It means the margin used for the right image, is also applied to the left. It implies more data processed, and more memory used.

:raw-html:`<h1>Implementation</h1>`

The Resampling application is implemented in the file ``cars/applications/dense_matching/census_mccnn_sgm_app.py``. The applications generated a CarsDataset of type "array", containing all the tiles of the Disparity Map.
Every tile is computed in parallel, using the Orchestrator framework. The wrapper function used for each tile, generates a xarray Dataset containing the disparity map.

The margin is computed using disparity range grid.
This grid is computed with direct location on the DEM min and max, with a certain margin.
Once the grid is computed, followed by the margin computation, the computed margin  is given to the Resampling application.

The margin used are removed before returning the tile to CARS.
