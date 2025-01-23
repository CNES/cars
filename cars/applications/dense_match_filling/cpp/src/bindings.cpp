#include "dense_match_filling.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(dense_match_filling_cpp, m) {
    m.doc() = "cars's pybind11 dense match filling module";

    m.def("fill_disp_pandora", &fill_disp_pandora, "");
    m.def("find_valid_neighbors", &find_valid_neighbors, "");
}