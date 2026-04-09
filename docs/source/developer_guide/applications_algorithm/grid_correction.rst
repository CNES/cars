===============
Grid Correction
===============

:raw-html:`<h1>Method</h1>`

The grid correction application uses the sparse matches computed by the sparse matching application, to correct the initial epipolar grid.

We follow the following steps:

  * Interpolate the sparse matches to generate an imperfect epipolar grid, and where whe should land in sensor geometry.
  * Simulate perfect matches: find where it should have landed if we had perfect matches.
  * Compute residuals: difference between raw and perfect sensor position.
  * We estimate a correction model, with least square fit.=: polynomial coefficients are estimated
  * We apply the correction model to the initial epipolar grid, to get a corrected epipolar grid.


:raw-html:`<h1>Limits of the method</h1>`



We assume that the error varies smoothly across the image, which is not always the case, for instance with vibrations.
The correction model is then not adapted to correct such errors.