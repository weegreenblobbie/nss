#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include "sorted_container.hpp" // Assuming SortedContainer is in sorted_container.hpp

namespace py = pybind11;

using std::cout;

py::array_t<float> squash_cpp(py::array_t<float, py::array::c_style | py::array::forcecast> array_in,
                             int center_m, int center_n, int radius, int size) 
{
    const auto array = array_in.unchecked();

    const int width = 2 * radius + 1;
    const int s2 = size / 2;
    const int m0 = center_m - radius - s2;
    const int n0 = center_n - radius - s2;

    // cout << "center_m: " << center_m << ", center_n: " << center_n << "\n";
    // cout << "radius: " << radius << ", size: " << size << ", width: " << width << "\n";
    // cout << "m0: " << m0 << ", n0: " << n0 << "\n";

    auto output_array = py::array_t<float, py::array::c_style>(std::vector<size_t>{static_cast<size_t>(width), static_cast<size_t>(width)});
    auto output = output_array.mutable_unchecked();

    nss::SortedContainer window;

    int m = 0;
    int n = 0;
    int m_direction = 1;

    while (n < width) 
    {
        cout << "m,n: " << m << ", " << n << ": ";

        // Initially fill the window with pixel values from the array.
        if (m == 0 && n == 0)
        {
            cout << "push(:,:)\n";
            for (int i = 0; i < size; ++i) 
            {
                for (int j = 0; j < size; ++j) 
                {
                    window.push({m0 + i, n0 + j, array(m0 + i, n0 + j)});
                }
            }
        }
        // Sliding the window down one row.
        else if (m_direction == 1 && m > 0) 
        {
            cout << "pop(" << m0 + m - 1 << ", :), push(" << m0 + m + size - 1 << ", :)\n";
            for (int j = 0; j < size; ++j) 
            {
                // Pop off old values from the previous row above (-1).
                window.pop({m0 + m - 1, n0 + n + j, array(m0 + m - 1, n0 + n + j)});
            }
            for (int j = 0; j < size; ++j)
            {
                // Push on new values from the new row (size - 1).
                window.push({m0 + m + size - 1, n0 + n + j, array(m0 + m + size - 1, n0 + n + j)});
            }
        } 
        // Sliding the window up one row.
        else if (m_direction == -1 && m < width - 1)
        {
            cout << "pop(" << m0 + m + size << ", :), push(" << m0 + m << ", :)\n";
            for (int j = 0; j < size; ++j)
            {
                // Pop off the old values from the row below (+1).
                window.pop({m0 + m + size, n0 + n + j, array(m0 + m + size, n0 + n + j)});
            }
            for (int j = 0; j < size; ++j)
            {
                // Push on new values from the the row above (size + 1)
                window.push({m0 + m, n0 + n + j, array(m0 + m, n0 + n + j)});
            }
        }

        // Sliding the window to the right by one column.
        else if (n > 0)
        {
            cout << "pop(:," << n0 + n - 1 << "), push(:," << n0 + n + size - 1 << ")\n";
            for (int i = 0; i < size; ++i)
            {
                // Pop off old values from the column on the left (-1).
                window.pop({m0 + m + i, n0 + n - 1, array(m0 + m + i, n0 + n - 1)});
            }
            for (int i = 0; i < size; ++i)
            {
                // Push on new values from the column on the right (size - 1).
                window.push({m0 + m + i, n0 + n + size - 1, array(m0 + m + i, n0 + n + size - 1)});
            }
        }

        // Center pixel in the window.
        const int cm = center_m - radius + m;
        const int cn = center_n - radius + n;
        const auto pixel = array(cm, cn);

        if (window.empty()) 
        {
            output(m, n) = pixel;
        }
        else 
        {
            const float min_val = window.min_value();
            const float max_val = window.max_value();
            const float mag = max_val - min_val;

            // if (mag < 1e-7) 
            // {
            //     output(m, n) = 0.0;
            // } 
            // else 
            {
                output(m, n) = (pixel - min_val) / mag;
            }
        }

        m += m_direction;

        if (m == width || m == -1) 
        {
            m_direction *= -1;
            m += m_direction;
            n++;
        }
        cout << "--> m,n: " << m << ", " << n << "\n";
    }

    return output_array;
}

PYBIND11_MODULE(utils_cpp, m) 
{
    m.def("squash_cpp", &squash_cpp, "Squash function in C++");
}