==============================
Disparity a priori Computation
==============================

:raw-html:`<h1>Method</h1>`


The Dense Matching application exposes a method to compute disparity a priori to use as in input of the main dense matching process / Pandora.

A low resolution regular grid is generated, and used in dense matching wrapper to compute full resolution disparity grids for current tile.

To generate the low resolution disparity range grids, we do direct localisation on initial elevation DEM (could be DEM median, or user provided DEM), and we retrieve the pixel values of initial elation + DEM min + DEM max on the found location.
Then we divide Z max - Z and Z - Z min by b/h ratio, to get the disparity range for each pixel of the grid.

:raw-html:`<h1>Limits of the method</h1>`

We make the hypothesis that Z, Z min and Z max are on the same pixel for a lign of sight, which is not always the case (not in Nadir)

The resolution of the generated grid must me chosen accordingly to dsm resolution, and filter size used for DEM min and max generation. If the grid is too coarse, the disparity range will be overestimated, and more memory will be used in Pandora. If the grid is too fine, the disparity range will loose some details.

