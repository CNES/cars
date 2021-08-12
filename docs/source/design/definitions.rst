================
CARS Definitions
================

This document contains CARS naming definitions to stabilize a common understanding in CARS development.

These definitions are intended to go in glossary when this future design is inserted in main CARS documentation.
Please update this part with all terms that need to be well defined.

TODO: extend this definition file for glossary and data structures definition (be careful to separate the conceptual defintions/requirements and the technical solution (class, ...))

General concepts definition
---------------------------

- pipeline: a set of chained 3D functions (example compute_dsm).
- step: a functional block (ie epipolar rectification) on one tile only
- tile: a subset of the entire input image to be processed on one worker only

Input data definitions
----------------------

- left_img: first image in a input stereo image pair (rename img1?): only internal maybe?
- right_img: second image in a input stereo image pair (rename img2?): only internal maybe?
- img1: first left image for external user interface (left is not understable by user)
- img2: second right image for external user interface (same)
- model1: geometric model for img1
- model2: geometric model for img2
- model_right: right geometric model.
- model_left: left geometric model.
- model_type: model type


Steps definitions
-----------------

- matches: generic term for absolute or relative matches: can be sparse matching or dense disparity map.
- matches_types: can be "sparse" or "dense" (TODO: to define !)

Modularity definitions
----------------------

- plugin: a modular possibility to add steps in 3D pipelines. Dependent where the functions are added.
- plugin_type: the type of plugin (for instance cloud filtering, disparity map filtering)
- loader: a modular possibility to load several libraries (example geometric libraries)
- loader_type: The type of the loader (for instance, geometric, matching, ...)
