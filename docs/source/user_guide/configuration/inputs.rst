.. _configuration_inputs:

======
Inputs
======

+-------------------------------------------------------------------------------------------+-----------------------+----------------------+----------+
| Name                | Description                                                         | Type                  | Default value        | Required |
+=====================+=====================================================================+=======================+======================+==========+
| *sensor*            | Stereo sensor images                                                | See next section      | No                   | Yes      |
+---------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
| *pairing*           | Association of image to create pairs                                | list of *sensor*      | No                   | Yes      |
+---------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
| *epsg*              | EPSG code                                                           | int, should be > 0    | None                 | No       |
+---------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
| *initial_elevation* | Field contains the path to the folder in which are located          | string                | None                 | No       |
|                     | the srtm tiles covering the production                              |                       |                      |          |
+---------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
| *default_alt*       | Default height above ellipsoid when there is no DEM available       | int                   | 0                    | No       |
|                     | no coverage for some points or pixels with no_data in the DEM tiles |                       |                      |          |
+---------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
| *roi*               | DSM roi file or bouding box                                         | string, list or tuple | None                 | No       |
+---------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
| *check_inputs*      | Check inputs consistency (to be deprecated and changed)             | Boolean               | False                | No       |
+---------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
| *geoid*             | geoid path                                                          | string                | Cars internal geoid  | No       |
+---------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+


.. _sensor:

Sensor
******

For each sensor images, give a particular name (what you want):

.. sourcecode:: text

    {
      "my_name_for_this_image":
        {
            "image" : "path_to_image.tif",
            "color" : "path_to_color.tif",
            "mask" : "path_to_mask.tif",
            "mask_classes" : {...}
            "nodata": 0
        }
    }


+-------------------+------------------------------------------------------------------------------------------+----------------+---------------+----------+
| Name              | Description                                                                              | Type           | Default value | Required |
+===================+==========================================================================================+================+===============+==========+
| *image*           | Path to the image                                                                        | string         |               | Yes      |
+-------------------+------------------------------------------------------------------------------------------+----------------+---------------+----------+
| *color*           | image stackable to image used to create an ortho-image corresponding to the produced dsm | string         |               | No       |
+-------------------+------------------------------------------------------------------------------------------+----------------+---------------+----------+
| *no_data*         | no data value of the image                                                               | int            | -9999         | No       |
+-------------------+------------------------------------------------------------------------------------------+----------------+---------------+----------+
| *geomodel*        | geomodel associated to the image                                                         | string         |               | Yes      |
+-------------------+------------------------------------------------------------------------------------------+----------------+---------------+----------+
| *geomodel_filters*| filters associated to the geomodel                                                       | List of string |               | No       |
+-------------------+------------------------------------------------------------------------------------------+----------------+---------------+----------+
| *mask*            | external mask of the image                                                               | string         | None          | No       |
+-------------------+------------------------------------------------------------------------------------------+----------------+---------------+----------+
|*mask_classes*     | mask's classes usage (see next section for more details)                                 | dict           |               | No       |
+-------------------+------------------------------------------------------------------------------------------+----------------+---------------+----------+

.. note::
    - *color*: This image can be composed of XS bands in which case a PAN+XS fusion will be performed.
    - If the *mask* is a multi-classes one and no *mask_classes*  configuration file is indicated, all non-zeros values of the mask will be considered as unvalid data.
    - The value 255 is reserved for CARS internal use, thus no class can be represented by this value in the masks.


CARS mask multi-classes structure
---------------------------------

Multi-classes masks have a unified CARS format enabling the use of several mask information into the API.
The classes can be used in different ways depending on the tag used in the dict defined below.

Dict is given in the *mask_classes* fields of sensor (see previous section).
This dict indicate the masks's classes usage and is structured as follows :

.. sourcecode:: text

    {
        "ignored_by_correlation": [1, 2],
        "set_to_ref_alt": [1, 3, 4],
        "ignored_by_sift_matching": [2]
    }


* The classes listed in *ignored_by_sift_matching* will be masked at the sparse matching step.
* The classes listed in *ignored_by_correlation* will be masked at the correlation step.
* The classes listed in *set_to_ref_alt* will be set to the reference altitude (srtm or scalar). To do so, these pixels's disparity will be set to 0.