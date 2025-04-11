#include <set>
#include <stdexcept>
#include <string>

namespace nss
{

std::out_of_range make_out_of_range(const char* file, int line, const std::string& message);

struct Pixel {
    int m;
    int n;
    float value;

    // Sort by 'value'.
    bool operator<(const Pixel& other) const
    {
        return value < other.value;
    }

    // Equality by exactly matching m, n.
    bool operator==(const Pixel& other) const
    {
        return m == other.m && n == other.n;
    }
};

class SortedContainer 
{
public:
    void push(const Pixel & pixel);
    void pop(const Pixel & pixel);
    // void push(float pixel);
    // void pop(float pixel);
    float min_value() const;
    float max_value() const;
    bool empty() const;
private:
    std::multiset<Pixel> _data;
    // std::multiset<float> _data;
};

} // namespace