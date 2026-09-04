.. _depth_to_z_fusion_app:

Depth to Z Fusion
=================

**Name**: "depth_to_z_fusion"

**Description**

Fuse monocular depth predictions onto stereo Z through spatial smoothing.
This application improves Z estimates by leveraging additional depth information (e.g., from trained depth models)
while preserving attachment to the original stereo triangulation.
The fusion solves an energy minimization problem that balances edge-aware smoothness with data fidelity.

This application runs after point cloud outlier removal steps and before point cloud refinement.
It operates on the per-tile surface at final resolution, producing a fitted depth map on the domain defined by depth input coverage.

**Configuration**

+------------------------------+----------------------------------------------------------+---------+-----------------------------------+------------------+----------+
| Name                         | Description                                              | Type    | Available value                   | Default value    | Required |
+==============================+==========================================================+=========+===================================+==================+==========+
| method                       | Method for depth to Z fusion                             | string  | "anisotropic"                     | "anisotropic"    | No       |
+------------------------------+----------------------------------------------------------+---------+-----------------------------------+------------------+----------+
| activated                    | Run this application (if false, skip processing)         | boolean |                                   | true             | No       |
+------------------------------+----------------------------------------------------------+---------+-----------------------------------+------------------+----------+
| save_intermediate_data       | Save fitted depth map and diagnostic layers as TIF       | boolean |                                   | false            | No       |
+------------------------------+----------------------------------------------------------+---------+-----------------------------------+------------------+----------+

If method is *anisotropic*:

+------------------------------+--------------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+
| Name                         | Description                                                                                                  | Type    | Available value                   | Default value | Required |
+==============================+==============================================================================================================+=========+===================================+===============+==========+
| lambda_data                  | Weight balancing data attachment (Z) vs. smoothness; higher -> closer to original Z                          | float   | >= 0                              | 0.02          | No       |
+------------------------------+--------------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+
| depth_sigma                  | Edge-preservation scale. Edges sharper than this are preserved; None -> per tile auto-estimate (recommended) | float   | > 0 or None                       | None          | No       |
+------------------------------+--------------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+
| depth_sigma_auto             | Auto-estimate depth_sigma from local depth statistics if depth_sigma is None                                 | boolean |                                   | true          | No       |
+------------------------------+--------------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+
| detail_scale                 | Gain on the depth-slope relief (1.0 = match depth slopes at full strength, >1 = stronger, <1 = weaker)       | float   | >= 0                              | 1.0           | No       |
+------------------------------+--------------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+
| cross_tile_weight            | Weight for cross-tile boundary constraints (higher -> smoother tile boundaries)                              | float   | >= 0                              | 1.0           | No       |
+------------------------------+--------------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+
| fill_values                  | None = keep Z unchanged, "invalid" = fill invalid Z only, "all" = replace all Z values                       | string  | None, "invalid", "all"            | "invalid"     | No       |
+------------------------------+--------------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+
| iterations                   | Maximum Jacobi solver iterations                                                                             | integer | > 0                               | 50            | No       |
+------------------------------+--------------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+
| tol                          | Convergence tolerance for Jacobi solver                                                                      | float   | > 0                               | 1.0e-4        | No       |
+------------------------------+--------------------------------------------------------------------------------------------------------------+---------+-----------------------------------+---------------+----------+

**Outputs**

The fused surface is saved with the following bands:

- ``fit_depth_map.tif``: Fitted Z from depth-guided fusion (defined on depth domain, with Z anchor subset)
- ``depth_to_z_residual.tif``: Difference (Z - fitted Z) at Z anchor points; diagnostic only
- ``depth_to_z_weight_map.tif``: Per-pixel smoothness weight from edge structure; diagnostic only
- ``filling.tif``: Filling provenance map (``EPI_FILLING``). Depth-to-Z updates the ``depth_to_z`` filling band where Z values were replaced.

By default (``fill_values: "invalid"``), valid stereo Z values are preserved and only invalid Z values are filled from ``fit_depth_map``.
If ``fill_values: "all"``, all Z values are replaced by fitted values.
If ``fill_values: null``, no replacement is applied.

**Example**

.. include-cars-config:: ../../example_configs/configuration/applications_depth_to_z_fusion
