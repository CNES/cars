===============
User interfaces
===============

Two levels:

- A main high level pipeline user level: run CARS pipeline with a possible configuration. From command line or Python API. Automatic mode.
- A specific low level 3D API with all functions exposed: works by tile. Advanced mode.


CARS Main High level pipeline API
=================================


Main High level scenarii
------------------------

Here are several main high level scenarii:

- Launch CARS with only one or several stereo images and geometric models pairs from CLI parameters and generate a DSM.

.. code-block:: console

    cars pair1.img_left=image_left.tif pair1.img_right=image_right.tif

- Identically, launch CARS with only one or several stereo images and geometric models pairs from CLI file and generate a DSM.

.. code-block:: console

    echo "pair1.img_left=image_left.tif pair1.img_right=image_right.tif" > config.yaml

    cars -i config.yaml # Example with yaml

- Identically, Launch CARS with only one or several stereo images and geometric models pairs from Python API and generate a DSM.

.. code-block:: console

    import cars
    param_list=["pair1.img_left=image_left.tif", "pair1.img_right=image_right.tif"]
    cars.run(param_list)


- Launch CARS prepare sub-pipeline only
- Launch CARS compute_dsm sub-pipeline only
- Launch CARS with any internal parameters from CLI
- Launch CARS with any internal parameters from Python API



CARS Main Python API
--------------------

1. cars.run(): Main Python API for cars. Parameters:

    - inputs: 1 pair only?, N pairs, N images?, mask, srtm,? One format in a class?
    - output: default: Default local directory. point_cloud?, dsm? an output.json file describing several elements?
    - configuration: parameters of each step, core libs, ... kwargs?

2. Sub pipelines

  2a. Prepare pipeline

.. code-block:: console

    cars.run(["prepare", "img_left=", "img_right=")
    or
    cars.run -i config.yaml
    or
    cars.pipelines.prepare.run()

  2b. Compute pipeline

.. code-block:: console

    cars.run(["compute", "img_left=", "img_right=")
    or
    cars.run -i config.yaml
    or
    cars.pipelines.compute.run()

.. note::

  Prepare and compute have the same API

Questions:

- Inputs format: a json file / Input format Class? Yaml --> if possible multi possibities (see cars_conf.rst)
- Output format: not only an output directory but also a json file / Output format class? the same than input incremented? Can be a default one configurable?
- Configuration: separation static_conf and dynamic parameters? have only one dynamic and static conf? --> see cars_conf.rst
- Rename compute_dsm in compute (not only dsm)

CARS Command Line Interface
---------------------------

The command line interface could be only corresponding to CARS Main Python API one to one

- cars -i config.yaml or conf.json ...
- cars pair1.img_left pair1.img_right

2. cars prepare pair1.img_left pair1.img_right # only prepare

3. cars compute pair1.img_left pair1.img_right #only compute


CARS 3D Functional User interfaces
==================================

This API is for a user to be able to launch CARS 3D specific algorithms for study, debug, test, ...


User scenarii
-------------

- Be able to open an image into a dataset

- Be able to launch image resample:

.. code-block:: console

    import cars.core.inputs
    import cars.core.geometry
    import cars.steps.rectification

    img_dataset = inputs.open(img_path)
    img_geom = geometry.open(img_geom_path??)

    resampled_dataset = rectification.resample_image(img_dataset, img_geom, grid_dataset, largest_size, region, nodata, mask)

TODO: Update when an evolution is done.

Steps
-----
Question: A step is a standardized API function in cars/steps?
TODO: Definition to finish, clarify

1.  rectification or generate_grid?

2.  matching:

    a. sparse_matching a step?
    b. dense_matching a step?

3. triangulate: which level?
4. filter point_cloud? plugin?
5. rasterize


Questions:

- Prepare steps AND compute steps? or only compute pipeline?
