.. include:: ../links_substitution.rst


CARS pipeline
=============

To summarize, CARS's default pipeline is organized into sequential steps, starting from input pairs (and metadata) and producing output data. Each step is performed tile-wise and distributed among workers. Part of the pipeline operates at multiple spatial resolutions to reach the final results while minimizing computation time. It is possible to run CARS at a single resolution, but this may be very inefficient for large or medium size images.


.. figure:: ../images/cars_pipeline_multi_pair.png
    :width: 1000px
    :align: center


The pipeline will perform the following steps |cars_isprs| |cars_igarss|:

- For each stereo pair:
    
    1. Create stereo-rectification grids for left and right views.
    2. Resample both images into epipolar geometry.
    3. Compute sift matches between left and right views in epipolar geometry.
    4. Create a bilinear correction model of the right image's stereo-rectification grid in order to minimize the epipolar error. Apply the estimated correction to the right grid.

- For each resolution (first_resolution, intermediate_resolution, last_resolution)
    
    - For each stereo pair:
        
        5. Resample the stereo pair in epipolar geometry, at the specified resolution using:
        
        	- The input :term:`DTM` (such as an SRTM) for the first resolution.
        	- The DEM Median from the previous resolution for intermediate_resolution or last_resolution.
        6. Compute the disparity map in epipolar geometry, by using the `DEM Min` and `DEM Max` as disparity intervals. For the first resolution, sift features are used to refine the disparity intervals.
        7. Triangulate the matches and get for each pixel of the reference image a latitude, longitude and altitude coordinate.

    - Then

        8. Merge points clouds coming from each stereo pairs.
        9. Filter the resulting 3D points cloud via two consecutive filters: the first removes the small groups of 3D points, the second filters the points which have the most scattered neighbors.
        10. Rasterize: Project these altitudes on a regular grid to create a `DSM` and its associated color, as well as (if not at the last resolution) `DEM Min/Max/Median`.
