# FeedforwardNT

### Description:

This program demonstrates the implementation of a neural network. It uses a stochastic gradient descent method with back propagation and a cross-entropy cost function. The activation function is the sigmoid function.

* Language: C
* Compiler: Clang/LLVM
* Platform: masOS/Linux (posix threads)
* Required library: BLAS/LAPACK

Compilation:

```
cd src
make depend
make
```

For the Iris data set, run with:

```
./FeedforwardNT ../params/parameters_iris.dat
```
For the MNIST data set, run with:

```
./FeedforwardNT ../params/parameters_mnist.dat
```

The file given to the program stores all (hyper)parameters needed for the run.

Notes:

1. An implementation of BLAS/LAPACK is required to compile the program.

2. The neural training code is not dependent on any particular input data set but currently it can only use the Iris and the MNIST data sets as input.

3. Metal acceleration will be implemented later and can be activated with -metal 

```
./FeedforwardNT ../params/parameters.dat -metal
```
4. One can specify a sperate data set to test the network. In that case, an validation set is also created from the training data set. If no test data are provided, they will be created from the training data set.

```
./FeedforwardNT ../params/parameters.dat --test-data path/to/file
```

5. Currently OpenCL is used to expreiment with offloading inference to the GPU. This is really just an experiment since the current implementation is not efficient because the network list traversal is done on the CPU. This implies that the code needs to call the GPU kernel at each layer of the network and to copy the computed activations to the next layer (i.e., copy between GPU buffers). Also some data need to be copied to and from the GPU for each data point of a data set. 
    
    Some of these limitations are partly due to the old OpenCL 1.2 which I use here although one can do a more efficient implementation with it albeit with the cost of having less clear data structures. Shared virtual memory in OpenCL 2.0 is a solution to that because it allows the use of pointer-linked data structures like linked lists or trees that are shared between the host and a device side of an OpenCL application. The GPU code is not activated by default, to compile the program with it, give -DUSE_OPENCL_GPU to CFLAGS in the Makefile.
