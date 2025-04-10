#include <set>

class SortedContainer 
{
public:
    void push(float x);
    void pop(float x);
    float min_value() const;
    float max_value() const;
    bool empty() const;
private:
    std::multiset<float> _data;
};