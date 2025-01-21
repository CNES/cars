#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <optional>

namespace py = pybind11;

py::array_t<bool> estimate_right_classif_on_left(
    py::array_t<bool> right_classif,
    py::array_t<float> disp_map,
    std::optional<py::array_t<bool>> disp_mask,
    int disp_min,
    int disp_max
) {
    auto r_right_classif = right_classif.unchecked<3>();
    auto r_disp_map = disp_map.unchecked<2>();

    size_t n_bands = r_right_classif.shape(0);
    size_t n_row = r_right_classif.shape(1);
    size_t n_col = r_right_classif.shape(2);

    bool use_disp_mask = disp_mask.has_value();
    std::unique_ptr<py::detail::unchecked_reference<bool, 2>> r_disp_mask;
    if (use_disp_mask) {
        r_disp_mask = std::make_unique<py::detail::unchecked_reference<bool, 2>>(
            disp_mask.value().unchecked<2>()
        );
    }

    py::array_t<bool> left_from_right_classif = py::array_t<bool>({n_bands, n_row, n_col});
    auto rw_left_from_right_classif = left_from_right_classif.mutable_unchecked<3>();

    for (size_t row = 0; row < n_row; row++) {
        for (size_t col = 0; col < n_col; col++) {

            // find classif
            float fdisp = r_disp_map(row, col);
            bool valid = use_disp_mask ? r_disp_mask->operator()(row, col) : !std::isnan(fdisp);

            if (valid) {
                // direct value
                int disp = static_cast<int>(std::floor(fdisp));
                
                for (size_t band = 0; band < n_bands; band++) {
                    rw_left_from_right_classif(band, row, col) = r_right_classif(band, row, col+disp);
                }

                continue;
            }

            // else: estimate with global range
            for (size_t band = 0; band < n_bands; band++) {
                
                bool found = false;
                for (
                    size_t col_classif = std::max(0, static_cast<int>(col)+disp_min);
                    col_classif < std::min(static_cast<int>(n_col), static_cast<int>(col)+disp_max);
                    col_classif++
                ) {
                    if (r_right_classif(band, row, col_classif)) {
                        found = true;
                        break;
                    }
                }

                rw_left_from_right_classif(band, row, col) = found;
            }

        }
    }
    return left_from_right_classif;
}

py::array_t<bool> mask_left_classif_from_right_mask(
    py::array_t<bool> left_classif,
    py::array_t<bool> right_mask,
    py::array_t<int> disp_min,
    py::array_t<int> disp_max
) {

    auto rw_left_classif = left_classif.mutable_unchecked<3>();
    auto r_right_mask = right_mask.unchecked<2>();
    auto r_disp_min = disp_min.unchecked<2>();
    auto r_disp_max = disp_max.unchecked<2>();

    size_t n_bands = rw_left_classif.shape(0);
    size_t n_row = rw_left_classif.shape(1);
    size_t n_col = rw_left_classif.shape(2);

    for (size_t row = 0; row < n_row; row++) {
        for (size_t col = 0; col < n_col; col++) {

            // estimate with global range
            bool all_masked = true;
            size_t lower_bound = std::max(0, static_cast<int>(col)+r_disp_min(row, col));
            size_t upper_bound = std::min(static_cast<int>(n_col), static_cast<int>(col)+r_disp_max(row, col));
            for (size_t col_classif = lower_bound; col_classif < upper_bound; col_classif++) {
                if (!r_right_mask(row, col_classif)) {
                    all_masked = false;
                    break;
                }
            }
            
            if (all_masked) {
                // Remove classif
                for (size_t band = 0; band < n_bands; band++) {
                    rw_left_classif(band, row, col) = false;
                }
            }
        }
    }

    return left_classif;
}
