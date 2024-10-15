# aligator-bench

This is a set of small benchmarks for the [aligator](https://github.com/Simple-Robotics/aligator) optimal control library.

## Contents

We test the following solvers:

- aligator's `SolverProxDDP`
- ALTRO (using a fork included as a submodule)
- the generic NLP solver [Ipopt](https://coin-or.github.io/Ipopt/)

## Building

**Dependencies** Building this repo requires:

- [aligator](https://github.com/Simple-Robotics/aligator)
- [gtest](https://github.com/google/googletest) | [conda](https://anaconda.org/conda-forge/gtest)
- [ipopt](https://coin-or.github.io/Ipopt/) | [repo](https://github.com/coin-or/Ipopt) | [conda](https://anaconda.org/conda-forge/ipopt)

These dependencies can easily be installed from conda/mamba:

```bash
mamba install -c conda-forge aligator gtest
```

The first step, as always, is to checkout the repository (recursively, as to get the submodules)

```bash
git clone https://github.com/Simple-Robotics/aligator-bench --recursive
```

Then, create the build dir and build away:

```bash
mkdir build && cd build
cmake ..  # with your usual options e.g. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build . -j<num-jobs>
```
