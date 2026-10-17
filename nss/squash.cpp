#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace py = pybind11;

using std::cout;

void
squash_cpp(
    py::array_t<float, py::array::c_style | py::array::forcecast> & array_in,
    int center_m,
    int center_n,
    int radius,
    int size) 
{
    auto & array = array_in.mutable_unchecked();

    const int width = 2 * radius + 1;
    const int s2 = size / 2;
    const int m0 = center_m - radius - s2;
    const int n0 = center_n - radius - s2;
    const std::vector<std::size_t> shape {
        static_cast<std::size_t>(width),
        static_cast<std::size_t>(width)
    };

    auto output_array = py::array_t<float, py::array::c_style>(shape);
    auto output = output_array.mutable_unchecked();

    for (int n = 0; n < width; ++n)
    {
        for (int m = 0; m < width; ++m)
        {
            // Center pixel in the window.
            const int cm = m0 + m + s2;
            const int cn = n0 + n + s2;

            const auto pixel = array(cm, cn);

            // If the pixel in the middle is nan, no action is needed.
            if (std::isnan(pixel)) continue;

            float max_val = std::nanf("");
            float min_val = std::nanf("");

            // Find the min and max pixel values in this sub-window.
            for (int i = 0; i < size; ++i)
            {
                for (int j = 0; j < size; ++j)
                {
                    const float pix = array(m0 + m + i, n0 + n + j);
                    if (!std::isnan(pix))
                    {
                        if (std::isnan(max_val)) max_val = pix;
                        if (std::isnan(min_val)) min_val = pix;
                        if (pix > max_val) max_val = pix;
                        if (pix < min_val) min_val = pix;
                    }
                }
            }

            const float mag = max_val - min_val;

            if (mag < 1e-7) 
            {
                output(m, n) = 0.0;
            } 
            else 
            {
                output(m, n) = (pixel - min_val) / mag;
            }
        }
    }

    // Write output back into array.
    const int m0_out = center_m - radius;
    const int n0_out = center_n - radius;
    for (int n = 0; n < width; ++n)
    {
        for (int m = 0; m < width; ++m)
        {
            array(m0_out + m, n0_out + n) = output(m, n);
        }
    }

    // return output_array;
}

PYBIND11_MODULE(utils_cpp, m) 
{
    m.def("squash_cpp", &squash_cpp, "Squash function in C++");
}