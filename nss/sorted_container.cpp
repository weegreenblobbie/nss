#include <cmath>
#include <stdexcept>

#include "sorted_container.hpp"

void
SortedContainer::push(float x)
{
    if (std::isnan(x)) return;
    _data.insert(x);
}

void
SortedContainer::pop(float x) 
{
    if (std::isnan(x)) return;

    auto it = _data.find(x);
    if (it != _data.end()) 
    {
        _data.erase(it);
    } 
    else 
    {
        throw std::out_of_range("Value not found in container.");
    }
}

float
SortedContainer::min_value() const 
{
    if (_data.empty()) 
    {
        throw std::out_of_range("Container is empty.");
    }
    return *_data.begin();
}

float
SortedContainer::max_value() const 
{
    if (_data.empty()) 
    {
        throw std::out_of_range("Container is empty.");
    }
    return *_data.rbegin();
}

bool
SortedContainer::empty() const
{
    return _data.empty();
}
