#include "robot_spec.hpp"
#include <filesystem>
#include <iostream>
#include <optional>
#include <toml11/conversion.hpp>
#include <toml11/find.hpp>
#include <toml11/parser.hpp>

namespace aligator_bench {

template <typename T>
auto sesecnoval_inplace(const std::optional<T> &v1, std::optional<T> &v2) {
  v2 = v1.has_value() ? v1 : v2;
  std::cout << "joined values " << v1.value_or("none") << " and "
            << v2.value_or("none") << std::endl;
}

template <typename V, typename U>
void _ssnv_apply_self(V U::*mptr, U &self, const U &other) {
  sesecnoval_inplace(other.*mptr, self.*mptr);
}

struct spec_impl_data {
  std::optional<std::string> path;
  std::optional<std::string> urdf_subpath;
  std::optional<std::string> urdf_filename;
  std::optional<std::string> srdf_subpath;
  std::optional<std::string> srdf_filename;
  std::optional<std::string> ref_posture;
  std::optional<bool> free_flyer;

  spec_impl_data &join(const spec_impl_data &other) {
    _ssnv_apply_self(&spec_impl_data::path, *this, other);
    _ssnv_apply_self(&spec_impl_data::urdf_subpath, *this, other);
    _ssnv_apply_self(&spec_impl_data::urdf_filename, *this, other);
    _ssnv_apply_self(&spec_impl_data::srdf_subpath, *this, other);
    _ssnv_apply_self(&spec_impl_data::srdf_filename, *this, other);
    _ssnv_apply_self(&spec_impl_data::ref_posture, *this, other);
    _ssnv_apply_self(&spec_impl_data::free_flyer, *this, other);
    return *this;
  }
};
} // namespace aligator_bench
TOML11_DEFINE_CONVERSION_NON_INTRUSIVE(aligator_bench::spec_impl_data, path,
                                       urdf_subpath, urdf_filename,
                                       srdf_subpath, srdf_filename, ref_posture,
                                       free_flyer)

namespace aligator_bench {

const fs::path PACKAGE_DIRS_BASE =
    fs::path(EXAMPLE_ROBOT_DATA_MODEL_DIR) / "../..";

robot_spec loadRobotSpecFromToml(const std::string &fname,
                                 const std::string &key, bool verbose) {
  const toml::value data = toml::parse(fname);

  const spec_impl_data parent = toml::get<spec_impl_data>(data);
  const spec_impl_data child = toml::find<spec_impl_data>(data, key);
  spec_impl_data c2 = parent;
  c2.join(child);

  if (!c2.path)
    throw std::runtime_error("Robot TOML file contained no \"path\" key.");

  const fs::path path =
      fs::path(EXAMPLE_ROBOT_DATA_MODEL_DIR) / c2.path.value();

  if (!c2.urdf_filename.has_value())
    throw std::runtime_error(
        "Robot TOML file contained no \"urdf_filename\" key.");
  const bool has_srdf = c2.srdf_filename.has_value();

  robot_spec result{
      path,
      path / c2.urdf_subpath.value_or("robots") / c2.urdf_filename.value(),
      has_srdf
          ? path / c2.srdf_subpath.value_or("srdf") / c2.srdf_filename.value()
          : "",
      c2.ref_posture.value_or("standing"), c2.free_flyer.value_or(false)};

  if (verbose) {
    std::cout << "Loaded robot:\n";
    std::cout << " > URDF file " << result.urdfPath << '\n';
    std::cout << " > SRDF file " << result.srdfPath << std::endl;
    if (result.floatingBase)
      std::cout << "Robot has floating base." << std::endl;
  }
  return result;
}

} // namespace aligator_bench
