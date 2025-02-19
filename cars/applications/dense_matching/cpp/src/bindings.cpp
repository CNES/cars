#include "dense_matching.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(dense_matching_cpp, m) {
    m.doc() = "cars's pybind11 dense matching module"; // optional module docstring

    m.def("estimate_right_classif_on_left", &estimate_right_classif_on_left, "");
    m.def("mask_left_classif_from_right_mask", &mask_left_classif_from_right_mask, "");
    m.def("estimate_right_grid_disp_int", &estimate_right_grid_disp<int>, "");
    m.def("estimate_right_grid_disp_float", &estimate_right_grid_disp<float>, "");
}