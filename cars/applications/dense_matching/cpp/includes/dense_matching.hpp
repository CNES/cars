#ifndef DENSE_MATCHING_HPP
#define DENSE_MATCHING_HPP

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Estimate right classif on left image
 *
 * @param right_classif right classification
 * @param disp_map disparity map
 * @param disp_mask disparity mask
 * @param disp_min disparity min
 * @param disp_max disparity max
 * @return right classif on left image
 */
py::array_t<bool> estimate_right_classif_on_left(
    py::array_t<bool> right_classif,
    py::array_t<float> disp_map,  
    std::optional<py::array_t<bool>> disp_mask,
    int disp_min,
    int disp_max
);

/**
 * @brief Mask left classif with right mask
 *
 * @param left_classif left classification
 * @param right_mask right mask
 * @param disp_min disparity min
 * @param disp_max disparity max
 * @return masked left classif
 */
py::array_t<bool> mask_left_classif_from_right_mask(
    py::array_t<bool> left_classif,
    py::array_t<bool> right_mask,
    py::array_t<int> disp_min,
    py::array_t<int> disp_max
);

/**
 * @brief Estimate right grid disparities from left grid disparities
 *
 * @param disp_min_grid left disparity minimum grid
 * @param disp_max_grid left disparity maximum grid
 * @return pair of right grid minimum and maximum disparities
 */
template<typename T> std::pair<py::array_t<T>, py::array_t<T>> estimate_right_grid_disp(
    py::array_t<T> disp_min_grid,
    py::array_t<T> disp_max_grid
);

#endif  // DENSE_MATCHING_HPP
