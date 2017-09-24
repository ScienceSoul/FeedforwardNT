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

Run with:

```
./FeedforwardNT ../params/parameters.dat
```

The parameters.dat file stores all parameters needed for the run.

Notes:

1. Please note that an implementation of BLAS/LAPACK is required to compile the program.

2. The neural training code is not dependent on any particular input data set but currently the program can only parse the iris data set.

3. The code tries to experiment with a very simple parallelization implementation which uses threads to parallelize the calculations inside a mini batch. It is not beneficial for a small network but on bigger ones, I observed a speed up of 30%-40% to process the training data set, albeit with the disadvantage of some memory overhead. This multithreaded mode can be activated with the command

    ```
    ./FeedforwardNT ../params/parameters.dat -pthread
    ```


4. The code also tries to experiment with offloading inference to the GPU. This is really just an experiment since the current implementation is not efficient because the network list traversal is done on the CPU. This implies that the code needs to call the GPU kernel at each layer of the network and to copy the computed activations to the next layer (i.e., copy between GPU buffers). Also some data need to be copied to and from the GPU for each data point of a data set. 
    
    Some of these limitations are partly due to the old OpenCL 1.2 which I use here although one can do a more efficient implementation with it albeit with the cost of having less clear data structures. Shared virtual memory in OpenCL 2.0 is a solution to that because it allows the use of pointer-linked data structures like linked lists or trees that are shared between the host and a device side of an OpenCL application. The GPU code is not activated by default, to compile the program with it, give -DUSE_OPENCL_GPU to CFLAGS in the Makefile.
