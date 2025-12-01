.. _auxiliary_filling_app:

Auxiliary Filling
=================

**Name**: "auxiliary_filling"

**Description**

Fill in the missing values of the texture and classification by using information from sensor inputs 
This application replaces the existing `image.tif` and `classification.tif`.

The application retrieves texture and classification information by performing inverse location on the input sensor images. It is therefore necessary to provide the `sensors` category in `inputs` configuration in order to use this application. The pairing information is also required: when searching for texture information, the application will always look in the first sensor of the pair and then in the second, if no information for the given pixel is found in the first sensor. The final filled value of the pixel is the average of the contribution of each pair. The classification information is a logical OR of all classifications.

In `fill_nan` mode, only the pixels that are no-data in the auxiliary images that are valid in the reference dsm will be filled while in full mode all valid pixel from the reference dsm are filled.

If `use_mask` is set to `true`, the texture data from a sensor will not be used if the corresponding sensor mask value is false. If the pixel is masked in all images, the filled texture will be the average of the first sensor texture of each pair

When ``save_intermediate_data`` is activated, the folder ``dump_dir/auxiliary_filling`` will contain the non-filled texture and classification.

**Configuration**

+------------------------------+---------------------------------------------+---------+----------------------------------+----------------------------------+----------+
| Name                         | Description                                 | Type    | Available values                 | Default value                    | Required |
+==============================+=============================================+=========+==================================+==================================+==========+
| method                       | Method for filling                          | string  | "auxiliary_filling_from_sensors" | "auxiliary_filling_from_sensors" | No       |
+------------------------------+---------------------------------------------+---------+----------------------------------+----------------------------------+----------+
| activated                    | Activates the filling                       | boolean |                                  | false                            | No       |
+------------------------------+---------------------------------------------+---------+----------------------------------+----------------------------------+----------+
| mode                         | Processing mode                             | string  | "fill_nan", "full"               | false                            | No       |
+------------------------------+---------------------------------------------+---------+----------------------------------+----------------------------------+----------+
| use_mask                     | Use mask information from input sensors     | boolean |                                  | true                             | No       |
+------------------------------+---------------------------------------------+---------+----------------------------------+----------------------------------+----------+
| texture_interpolator         | Interpolator used for texture interpolation | string  | "linear", "nearest", "cubic"     | "linear"                         | No       |
+------------------------------+---------------------------------------------+---------+----------------------------------+----------------------------------+----------+
| save_intermediate_data       | Saves the temporary data in dump_dir        | boolean |                                  | false                            | No       |
+------------------------------+---------------------------------------------+---------+----------------------------------+----------------------------------+----------+
