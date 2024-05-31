# aligator-bench

This is a set of small benchmarks for the [aligator](https://github.com/Simple-Robotics/aligator) optimal control library.

## Building

**Dependencies** Building this repo requires:

- aligator
- [gtest](https://github.com/google/googletest)
- [benchmark](https://github.com/google/benchmark)


The first step, as always, is to checkout the repository (recursively, as to get the submodules)

```
git clone https://github.com/Simple-Robotics/aligator-bench --recursive
```

Then, create the build dir and build away:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release # with your usual options e.g. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build . -j<num-jobs>
```
