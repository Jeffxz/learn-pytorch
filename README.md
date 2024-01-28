# Learn Pytorch in C++

Wysebee Core is customized Pytorch for environment using c++ programing language.

## Goal

* Learn ML using pytorch and C++
* Compile for embedding system

## Build from source

### Build

```
mkdir build
cd build
cmake ../pytorch_src
make -j8
```

### Install

```
cd build
cmake --install . --prefix ../output
```

## Build examples

```
cd examples/torch/hello-torch/
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=<absolute path for output> ..
cmake --build .
```
