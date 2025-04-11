#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace py = pybind11;

using std::cout;

py::array_t<float> squash_cpp(py::array_t<float, py::array::c_style | py::array::forcecast> & array_in,
                             int center_m, int center_n, int radius, int size) 
{
    const auto array = array_in.unchecked();

    const int width = 2 * radius + 1;
    const int s2 = size / 2;
    const int m0 = center_m - radius - s2;
    const int n0 = center_n - radius - s2;

    auto output_array = py::array_t<float, py::array::c_style>(std::vector<size_t>{static_cast<size_t>(width), static_cast<size_t>(width)});
    auto output = output_array.mutable_unchecked();

    std::vector<float> window;
    window.reserve(size * size);

    int m = 0;
    int n = 0;
    int m_direction = 1;

    int last_n = -1;
    const std::string erase = "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";
    while (n < width) 
    {
        if (last_n != n && n  % 10 == 0)
        {
            cout << erase << n + 1 << "/" << width << ": squashing ..." << std::flush;
            last_n = n;
        }

        // Center pixel in the window.
        const int cm = center_m - radius + m;
        const int cn = center_n - radius + n;
        const auto pixel = array(cm, cn);

        if (std::isnan(pixel))
        {
            m += m_direction;
            if (m == width || m == -1) 
            {
                m_direction *= -1;
                m += m_direction;
                n++;
            }
            continue;
        }

        window.clear();

        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                float pix = array(m0 + m + i, m0 + n + j);
                if (!std::isnan(pix))
                {
                    window.push_back(pix);
                }
            }
        }

        float max_val = std::nanf("");
        float min_val = std::nanf("");

        if (!window.empty())
        {
            max_val = window[0];
            min_val = window[0];
        }

        for (const float & pixel : window)
        {
            if (pixel > max_val) max_val = pixel;
            if (pixel < min_val) min_val = pixel;
        }

        if (window.empty())
        {
            output(m, n) = pixel;
        }
        else
        {
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

        m += m_direction;

        if (m == width || m == -1) 
        {
            m_direction *= -1;
            m += m_direction;
            n++;
        }
    }

    return output_array;
}

PYBIND11_MODULE(utils_cpp, m) 
{
    m.def("squash_cpp", &squash_cpp, "Squash function in C++");
}