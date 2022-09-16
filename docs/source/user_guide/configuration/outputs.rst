.. _configuration_outputs:

=======
Outputs
=======

+----------------+-------------------------------------------------------------+--------+----------------+----------+
| Name           | Description                                                 | Type   | Default value  | Required |
+================+=============================================================+========+================+==========+
| out_dir        | Output folder where results are stored                      | string | No             | Yes      |
+----------------+-------------------------------------------------------------+--------+----------------+----------+
| dsm_basename   | base name for dsm                                           | string | "dsm.tif"      | No       |
+----------------+-------------------------------------------------------------+--------+----------------+----------+
| color_basename | base name for  ortho-image                                  | string | "color.tif     | No       |
+----------------+-------------------------------------------------------------+--------+----------------+----------+
| info_basename  | base name for file containing information about computation | string | "content.json" | No       |
+----------------+-------------------------------------------------------------+--------+----------------+----------+

Output contents
***************

The output directory, defined on the configuration file (see previous section) contains at the end of the computation:

* the dsm
* color image (if *color image* has been given)
* information json file containing: used parameters, information and numerical results related to computation, step by step and pair by pair.
* subfolder for each defined pair which can contains intermediate data