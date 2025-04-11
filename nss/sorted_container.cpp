#include <cmath>
#include <stdexcept>
#include <sstream>
#include <string>
#include <iostream>

#include "sorted_container.hpp"

namespace nss
{

void
SortedContainer::push(const Pixel & pixel)
{
    if (!std::isfinite(pixel.value)) return;

    std::cout << "    push(" << pixel.m << ", " << pixel.n << ")\n";
    _data.insert(pixel);
}

void
SortedContainer::pop(const Pixel & pixel) 
{
    std::cout << "    pop(" << pixel.m << ", " << pixel.n << ")";
    if (!std::isfinite(pixel.value))
    {
        std::cout << "\n";
        return;
    }
    auto it = std::find(_data.begin(), _data.end(), pixel);
    if (it != _data.end())
    {
        _data.erase(it);
        std::cout << " erased!\n";
    }
    else
    {
        std::stringstream ss;
        ss << "pop(" << pixel.m << ", " << pixel.n << "): Pixel wasn't found.";
        throw make_out_of_range(__FILE__, __LINE__, ss.str());
    }
}

float
SortedContainer::min_value() const 
{
    if (_data.empty()) 
    {
        throw make_out_of_range(__FILE__, __LINE__, "Container is empty.");
    }
    return _data.begin()->value;
}

float
SortedContainer::max_value() const 
{
    if (_data.empty()) 
    {
        throw make_out_of_range(__FILE__, __LINE__, "Container is empty.");
    }
    return _data.rbegin()->value;
}

bool
SortedContainer::empty() const
{
    return _data.empty();
}

std::out_of_range
make_out_of_range(const char* file, int line, const std::string& message)
{
    std::stringstream ss;
    ss << file << "(" << line << "): " << message;
    return std::out_of_range(ss.str());
}

} // nss namespace