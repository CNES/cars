================
CARS Definitions
================

This document contains CARS naming definition to stabilize a common understanding in CARS development.

- pipeline : a set of chained 3D functions (example compute_dsm).
- step: a functional block (ie epipolar rectification) on one tile only
- tile: a subset of the entire input image to be processed on one worker only

- left_img : first image in a input stereo image pair (rename img1?)
- right_img : second image in a input stereo image pair (rename img2?)


- plugin: a modular possibility to add steps in 3D pipelines. Dependent where the functions are added.
- plugin type: the type of plugin (for instance cloud filtering, disparity map filtering)
- loader: a modular possibility to load different libraries (example geometric libraries)
- loader type: The type of the loader (for instance, geometric, matching, ...)
