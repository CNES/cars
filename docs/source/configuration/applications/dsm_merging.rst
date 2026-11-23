.. _dsm_merging_app:

DSM merging
===========


**Name**: "dsm_merging"

**Description**

Merge multiple DSMs and their associated auxiliary data according to a priority order.

**Configuration**

+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------------------------+----------+
| Name                            | Description                                                              | Type       | Available value                      | Default value                            | Required |
+=================================+==========================================================================+============+======================================+==========================================+==========+
| method                          | Method for dsm_merging                                                   | string     | "weighted_fusion"                    | "weighted_fusion"                        | No       |
+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------------------------+----------+
| tile_size                       | Size of tiling                                                           | int        |                                      | 4000                                     | No       |
+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------------------------+----------+
| filling_priority_list           | The filling priority list                                                | list       |                                      | [6, 1, 4, 5, 3, 2, 7, 0]                 | No       |
+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------------------------+----------+
| classif_priority_list           | The classif priority list                                                | str        |                                      | [7, 8, 9, 5, 3, 1, 2, 4, 6, 10, 11, 255] | No       |
+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------------------------+----------+

Examples
~~~~~~~~

.. include-cars-config:: ../../example_configs/configuration/applications_dsm_merging

Priority rules
~~~~~~~~~~~~~~

The DSMs are merged according to a configurable priority order, which can be adjusted in the configuration file. If you request values that are not available in the corresponding filling or classification layers, CARS will fail with an error.

**filling_priority_list**

+---------------------------------+--------------------------------------------------------------------------+
| value                           | corresponding filling                                                    |
+=================================+==========================================================================+
| 6                               | fill_with_geoid                                                          |
+---------------------------------+--------------------------------------------------------------------------+
| 1                               | no_edition                                                               |
+---------------------------------+--------------------------------------------------------------------------+
| 4                               | fill_with_endogenous_dem                                                 |
+---------------------------------+--------------------------------------------------------------------------+
| 5                               | interpolate_from_borders                                                 |
+---------------------------------+--------------------------------------------------------------------------+
| 3                               | interpolation                                                            |
+---------------------------------+--------------------------------------------------------------------------+
| 2                               | fill_with_exogenous_dem                                                  |
+---------------------------------+--------------------------------------------------------------------------+
| 7                               | other                                                                    |
+---------------------------------+--------------------------------------------------------------------------+
| 0                               | no_data                                                                  |
+---------------------------------+--------------------------------------------------------------------------+

Following this priority list, only the DSMs and their associated auxiliary data corresponding to the selected filling classes will be merged.

**classif_priority_list (using slurp)**

+---------------------------------+--------------------------------------------------------------------------+
| value                           | corresponding classification                                             |
+=================================+==========================================================================+
| 7                               | sea                                                                      |
+---------------------------------+--------------------------------------------------------------------------+
| 8                               | lake                                                                     |
+---------------------------------+--------------------------------------------------------------------------+
| 9                               | river                                                                    |
+---------------------------------+--------------------------------------------------------------------------+
| 5                               | cloud                                                                    |
+---------------------------------+--------------------------------------------------------------------------+
| 3                               | water                                                                    |
+---------------------------------+--------------------------------------------------------------------------+
| 1                               | low_vegetation                                                           |
+---------------------------------+--------------------------------------------------------------------------+
| 2                               | high_vegetation                                                          |
+---------------------------------+--------------------------------------------------------------------------+
| 4                               | buildings                                                                |
+---------------------------------+--------------------------------------------------------------------------+
| 6                               | bare_ground                                                              |
+---------------------------------+--------------------------------------------------------------------------+
| 10                              | false_positive_buildings                                                 |
+---------------------------------+--------------------------------------------------------------------------+
| 11                              | background                                                               |
+---------------------------------+--------------------------------------------------------------------------+
| 255                             | no_data                                                                  |
+---------------------------------+--------------------------------------------------------------------------+

For the classification, several classification values may be available either for the same filling class or in the case of no_edition. Therefore, The majority classification is selected, and a priority list is used to resolve ties when multiple classifications have the same number of occurrences.


Overlap difference statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During DSM merging, elevation difference statistics are computed incrementally,
before each new DSM is merged with the previously merged DSMs.

For each new DSM, the elevation differences are calculated as:

.. math::

    \Delta Z = Z_{\text{new DSM}} - Z_{\text{previously merged DSMs}}

Only pixels with valid elevations and non-zero weights in both DSMs are
considered. The first DSM has no statistics, as no previous DSM is available
for comparison.

Statistics are computed per tile and then aggregated to obtain global
statistics for each merging step. The aggregation is weighted by the number
of valid overlapping pixels in each tile.

The following statistics are stored in the ``metadata.yaml`` file:

- ``compared_pixels``: total number of valid overlapping pixels.
- ``mean``: exact global mean of elevation differences, computed as the
  weighted mean of tile means.
- ``std``: exact global population standard deviation, reconstructed
  from tile means, standard deviations and pixel counts.
- ``median``: approximation of the global median, computed as the
  weighted mean of tile medians.

.. note::

    The reported median is not the exact median of all elevation differences.
    It is the weighted mean of tile medians, using the number of valid
    overlapping pixels in each tile as weights.

Only the aggregated statistics are saved in ``metadata.yaml``.
Per-tile statistics are not included in the final metadata.

Example of generated metadata:

.. code-block:: yaml

    merging:
      applications:
        dsm_merging:
          overlap_difference_statistics:
            unit: m
            inputs:
              dsm12:
                previous_merged_dsms: []
                overlap: null
              dsm13:
                previous_merged_dsms:
                  - dsm12
                overlap:
                  compared_pixels: 1675659
                  mean: 1.0604
                  median: 1.0111
                  std: 2.5354
              dsm23:
                previous_merged_dsms:
                  - dsm12
                  - dsm13
                overlap:
                  compared_pixels: 1703756
                  mean: 0.0419
                  median: 0.0524
                  std: 1.5458

The ``previous_merged_dsms`` field identifies the DSMs already merged
before the current DSM is added. The ``overlap`` field is ``null`` when
no valid comparison is available.

Elevation difference statistics are expressed in meters, assuming the
input DSM elevations are provided in meters.
