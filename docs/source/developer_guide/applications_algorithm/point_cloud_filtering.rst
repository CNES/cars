=====================
Point Cloud Filtering
=====================

:raw-html:`<h1>Method</h1>`

After the depth map (x, y, z in structured array in epipolar geometry) is generated in Triangulation application, we filter it to remove outliers.

2 methods are implemented:

  * statistical: using the k nearest neighbors, we compute the mean distance (or median distance), standard deviation, and remove points that are further than a defined threshold.
  * small component: clusters are generated, with defined radius from each point. Then we filter the clusters with a minimum number of points per cluster.

Both method are used in epipolar geometry. It means that the neighboring points are the neighboring pixels in the epipolar geometry. With this in consideration, we dont have to compute a kd-tree in order to accelerate neighbors computation.

:raw-html:`<h1>Limits of the method</h1>`


  * A group of outliers could be close to each other, and then not flagged as outliers with both methods.
  * The method is sensible to the window used to search neighbors. We expect the k nearest neighbors to be in the research window, whereas they could be outside of it, when disparity range are large, and disparity varies a lot between pixels.



:raw-html:`<h1>Implementation</h1>`

We use the cars-filter library to perform the filtering.