#include "dense_match_filling.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <optional>
#include <iostream>
#include <algorithm>

namespace py = pybind11;

std::pair<py::array_t<float>, py::array_t<bool>> fill_disp_pandora(
    py::array_t<float> disp,
    py::array_t<bool> msk_fill_disp,
    int nb_directions
) {
    auto r_disp = disp.unchecked<2>();
    auto r_msk_fill_disp = msk_fill_disp.unchecked<2>();

    size_t nrow = r_disp.shape(0);
    size_t ncol = r_disp.shape(1);

    py::array_t<float> out_disp({nrow, ncol});
    py::array_t<bool> out_msk({nrow, ncol});

    auto rw_out_disp = out_disp.mutable_unchecked<2>();
    auto rw_out_msk = out_msk.mutable_unchecked<2>();

    py::array_t<float> dirs;
    if (nb_directions == 8) {
        std::vector<float> data = {
            0.0f,  1.0f,
            -1.0f,  1.0f,
            -1.0f,  0.0f,
            -1.0f, -1.0f,
            0.0f, -1.0f,
            1.0f, -1.0f,
            1.0f,  0.0f,
            1.0f,  1.0f
        };

        dirs = py::array({8, 2}, data.data());
    } else if (nb_directions == 16) {
        std::vector<float> data = {
            0.0f,  1.0f, -0.5f,  1.0f,
            -1.0f,  1.0f, -1.0f,  0.5f,
            -1.0f,  0.0f, -1.0f, -0.5f,
            -1.0f, -1.0f, -0.5f, -1.0f,
            0.0f, -1.0f,  0.5f, -1.0f,
            1.0f, -1.0f,  1.0f, -0.5f,
            1.0f,  0.0f,  1.0f,  0.5f,
            1.0f,  1.0f,  0.5f,  1.0f
        };

        dirs = py::array({16, 2}, data.data());
    } else {
        throw std::invalid_argument("nb_directions must be 8 or 16");
    }

    for (size_t row = 0; row < nrow; ++row) {
        for (size_t col = 0; col < ncol; ++col) {
            if (r_msk_fill_disp(row, col)) {
                auto valid_neighbors = find_valid_neighbors(
                    dirs, disp, msk_fill_disp,
                    static_cast<int>(row), static_cast<int>(col),
                    nb_directions
                ).unchecked<1>();

                std::vector<float> valid_values;
                for (size_t i = 0; i < valid_neighbors.shape(0); ++i) {
                    if (!std::isnan(valid_neighbors(i))) {
                        valid_values.push_back(valid_neighbors(i));
                    }
                }


                if (valid_values.empty()) {
                    rw_out_disp(row, col) = std::nanf("");
                }
                else {
                    std::sort(valid_values.begin(), valid_values.end());
                    if (valid_values.size()%2==1) {
                        rw_out_disp(row, col) = valid_values[valid_values.size()/2];
                    } else {
                        rw_out_disp(row, col) = valid_values[valid_values.size()/2-1] 
                                              + valid_values[valid_values.size()/2];
                        rw_out_disp(row, col) /= 2.f;
                    }
                }
                rw_out_msk(row, col) = false;
            } else {
                rw_out_disp(row, col) = r_disp(row, col);
                rw_out_msk(row, col) = r_msk_fill_disp(row, col);
            }
        }
    }

    return {out_disp, out_msk};
}

py::array_t<float> find_valid_neighbors(
    py::array_t<float> dirs,
    py::array_t<float> disp,
    py::array_t<bool> valid,
    int row,
    int col,
    int nb_directions
) {
    auto r_dirs = dirs.unchecked<2>();
    auto r_disp = disp.unchecked<2>();
    auto r_valid = valid.unchecked<2>();

    size_t nrow = r_disp.shape(0);
    size_t ncol = r_disp.shape(1);
    size_t max_path_length = std::max(nrow, ncol);

    py::array_t<float> valid_neighbors(nb_directions);
    auto rw_valid_neighbors = valid_neighbors.mutable_unchecked<1>();

    for (int direction = 0; direction < nb_directions; ++direction) {

        rw_valid_neighbors(direction) = 0.f;

        for (size_t i = 1; i < max_path_length; ++i) {
            int tmp_row = row + static_cast<int>(r_dirs(direction, 0) * static_cast<float>(i));
            int tmp_col = col + static_cast<int>(r_dirs(direction, 1) * static_cast<float>(i));

            if (tmp_row < 0 || tmp_row >= static_cast<int>(nrow) ||
                tmp_col < 0 || tmp_col >= static_cast<int>(ncol)) {
                rw_valid_neighbors(direction) = std::nanf("");
                break;
            }

            if (!r_valid(tmp_row, tmp_col) && r_disp(tmp_row, tmp_col) != 0) {
                rw_valid_neighbors(direction) = r_disp(tmp_row, tmp_col);
                break;
            }
        }
    }

    return valid_neighbors;
}
