#include <aligator/python/utils.hpp>

namespace bp = boost::python;

inline bp::arg operator""_a(const char *name, std::size_t) {
  return bp::arg(name);
}
