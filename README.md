# aligator-bench

This is a set of small benchmarks for the [aligator](https://github.com/Simple-Robotics/aligator) optimal control library.

## Contents

## Building

**Dependencies** Building this repo requires:

- [aligator](https://github.com/Simple-Robotics/aligator)
- [gtest](https://github.com/google/googletest)
- [ipopt](https://github.com/coin-or/Ipopt) | [conda](https://anaconda.org/conda-forge/ipopt)

These dependencies can easily be installed from conda/mamba:

```bash
mamba install -c conda-forge aligator gtest benchmark
```

The first step, as always, is to checkout the repository (recursively, as to get the submodules)

```bash
git clone https://github.com/Simple-Robotics/aligator-bench --recursive
```

Then, create the build dir and build away:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release # with your usual options e.g. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build . -j<num-jobs>
```
