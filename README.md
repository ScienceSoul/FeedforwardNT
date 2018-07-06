# FeedforwardNT

```
NOTE: The code developed for this project is now used for the BrainStorm library. See the latter for further development. 
```

### Description:

This program demonstrates the implementation of a feedforward neural network. It uses a stochastic gradient descent method with back propagation and a cross-entropy cost function. The activation function is the sigmoid function.

* Language: C
* Compiler: Clang/LLVM
* Platform: masOS/Linux
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
./FeedforwardNT ../params/parameters.dat -test-data path/to/file
```
