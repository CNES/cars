.. _applications:

Applications
============

The `applications` key is optional and is used to redefine parameters for each application used in the pipeline. 
You can personnalize the configuration for each resolution at which the pipeline is ran, or override the parameters for all resolutions at once, as explained in the section right below. 

.. tabs::

    .. tab:: Overriding all resolutions at once

        This is the default behaviour when providing a configuration dict directly in the `applications` key.

        This example overrides the configuration of `application_name` for all resolutions at once :

        .. include-cars-config:: ../example_configs/configuration/applications_override_all_resolutions


    .. tab:: Overriding a single resolution

        To override a configuration at a specific resolution, you first need to identify which resolution you want to modify. By default, CARS uses the resolutions 16, 4, and 1 :
        
        - Resolution 16 corresponds to 16 times the original resolution (e.g., 16m if the original resolution is 1m).
        - Resolution 4 corresponds to 4 times the original resolution (e.g., 4m if the original resolution is 1m).
        - Resolution 1 corresponds to the original resolution (e.g., 1m).

        Once you have chosen the resolution value, you can override the configuration by adding an entry to the `applications` dictionary with the key `resolution_{resolution_value}` with resolution value an integer.

        The following example overrides the configuration for `application_name` at resolutions 4 and 1, using different parameters for each. Resolution 16 will retain its default configuration.

        .. include-cars-config:: ../example_configs/configuration/applications_override_single_resolution

The pages below describes all the available parameters for each CARS application.

CARS applications are defined and called by their **name**. An example configuration is provided for each application.

Be careful with these parameters: no mechanism ensures consistency between applications for now. Some parameters can degrade performance and DSM quality heavily.
The default parameters have been set as a robust and consistent end to end configuration for the whole pipeline.

.. toctree::
   :caption: Application parameters
   :maxdepth: 1

   applications/grid_generation
   applications/resampling
   applications/sparse_matching
   applications/dem_generation
   applications/ground_truth_reprojection
   applications/dense_matching
   applications/dense_match_filling
   applications/triangulation
   applications/point_cloud_outlier_removal
   applications/point_cloud_rasterization
   applications/dsm_filling
   applications/auxiliary_filling


The default configuration can be different for the first resolution, the intermediate resolution(s) and the last resolution. 
The changes to the default values can be modified in the source code, in ``cars/pipelines/conf_resolution/*``.

The section below includes the files directly.

.. tabs::

    .. tab:: Overriding configuration : first resolution

        This is empty for now.
        
        .. include:: ../../../cars/pipelines/conf_resolution/conf_first_resolution.json
            :literal:

    .. tab:: Overriding configuration : all intermediate resolutions
        
        This is empty for now.

        .. include:: ../../../cars/pipelines/conf_resolution/conf_intermediate_resolution.json
            :literal:

    .. tab:: Overriding configuration : final resolution
        
        .. include:: ../../../cars/pipelines/conf_resolution/conf_final_resolution.json
            :literal: