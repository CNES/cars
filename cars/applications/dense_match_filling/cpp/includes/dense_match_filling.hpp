#ifndef DENSE_MATCH_FILLING_HPP
#define DENSE_MATCH_FILLING_HPP

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Interpolate the left disparity map to fill holes.
 *        Invalid pixels are interpolated by finding the nearest valid pixels in
 *        8 or 16 directions and taking the median of their disparities.
 *
 * @param disp Disparity map (2D array: row, col)
 * @param msk_fill_disp Validity mask (2D array: row, col)
 * @param nb_directions Number of directions to explore (8 or 16)
 * @return Tuple containing the interpolated disparity map and updated validity mask
 */
std::pair<py::array_t<float>, py::array_t<bool>> fill_disp_pandora(
    py::array_t<float> disp,
    py::array_t<bool> msk_fill_disp,
    int nb_directions
);

/**
 * @brief Find valid neighbors along specified directions.
 *
 * @param dirs Directions to explore (2D array: direction x [row_offset, col_offset])
 * @param disp Disparity map (2D array: row, col)
 * @param valid Validity mask (2D array: row, col)
 * @param row Current row index
 * @param col Current column index
 * @param nb_directions Number of directions to explore
 * @return Array of valid neighbors’ disparities along each direction
 */
py::array_t<float> find_valid_neighbors(
    py::array_t<float> dirs,
    py::array_t<float> disp,
    py::array_t<bool> valid,
    int row,
    int col,
    int nb_directions
);

#endif  // DENSE_MATCH_FILLING_HPP
