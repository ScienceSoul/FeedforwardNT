//
//  NeuralNetwork.h
//  FeedforwardNT
//
//  Created by Seddik hakime on 31/05/2017.
//

#ifndef NeuralNetwork_h
#define NeuralNetwork_h

#ifdef USE_OPENCL_GPU
    #include "OpenCLUtils.h"
#endif

#include <stdio.h>
#include "Utils.h"
#include "TimeProfile.h"

#endif /* NeuralNetwork_h */

#ifdef USE_OPENCL_GPU

#define OPENCL_PROGRAM_FILE_LOC1 "./kernel/inference.cl"
#define OPENCL_PROGRAM_FILE_LOC2 "../kernel/inference.cl"

typedef struct gpuInference {
    int m;
    int n;
    cl_mem _Nullable W;
    cl_mem _Nullable A;
    cl_mem _Nullable B;
    cl_mem _Nullable Z;
    // The GPU kernel associated with the sgemv operation
    cl_kernel _Nullable kernel;
    struct gpuInference * _Nullable next;
    struct gpuInference * _Nullable previous;
} gpuInference;

typedef struct GPUCompute {
    struct gpuInference * _Nullable gpuInferenceStore;
    cl_program _Nullable program;
    cl_device_id _Nullable device;
    cl_context  _Nullable context;
    cl_command_queue _Nullable queue;
    
    // The sgemv routine
    void (* _Nullable inference)(void * _Nonnull self, gpuInference * _Nonnull gpuInferenceStore);
} GPUCompute;

 gpuInference * _Nonnull allocateGPUInference(void);
 GPUCompute * _Nonnull  allocateGPUCompute(void);
 void setUpOpenCLDevice(GPUCompute *compute);
 gpuInference * _Nonnull initGPUInferenceStore(GPUCompute *compute, weightNode * _Nonnull weightsList, activationNode * _Nonnull activationsList, int * _Nonnull ntLayers, size_t numberOfLayers);
void inference(void * _Nonnull self, gpuInference * _Nonnull gpuInferenceStore);

#endif

typedef struct weightNode {
    size_t m, n;
    float * _Nullable * _Nullable w;
    struct weightNode * _Nullable next;
    struct weightNode * _Nullable previous;
} weightNode;

typedef struct biasNode {
    size_t n;
    float * _Nullable b;
    struct biasNode * _Nullable next;
    struct biasNode * _Nullable previous;
} biasNode;

typedef struct activationNode {
    size_t n;
    float * _Nullable a;
    struct activationNode * _Nullable next;
    struct activationNode * _Nullable previous;
} activationNode;

typedef struct zNode {
    size_t n;
    float * _Nullable z;
    struct zNode * _Nullable next;
    struct zNode * _Nullable previous;
} zNode;

typedef struct dcdwNode {
    size_t m, n;
    float * _Nullable * _Nullable dcdw;
    struct dcdwNode * _Nullable next;
    struct dcdwNode * _Nullable previous;
} dcdwNode;

typedef struct dcdbNode {
    size_t n;
    float * _Nullable dcdb;
    struct dcdbNode * _Nullable next;
    struct dcdbNode * _Nullable previous;
} dcdbNode;

typedef struct NeuralNetwork {
    float * _Nullable * _Nullable batch;
    int example_idx;
    int number_of_features;
    
    size_t number_of_layers;
    size_t max_number_of_nodes_in_layer;
    
    weightNode * _Nullable weightsList;
    biasNode * _Nullable biasesList;
    activationNode * _Nullable activationsList;
    zNode * _Nullable zsList;
    dcdwNode * _Nullable dcdwsList;
    dcdbNode * _Nullable dcdbsList;
    dcdwNode * _Nullable delta_dcdwsList;
    dcdbNode * _Nullable delta_dcdbsList;
    
#ifdef USE_OPENCL_GPU
    GPUCompute * _Nullable compute;
#endif
    
    void (* _Nullable create)(void * _Nonnull self, int * _Nonnull ntLayers, size_t numberOfLayers, int * _Nullable miniBatchSize);
    void (* _Nullable destroy)(void * _Nonnull self);
    
    void (* _Nullable SDG)(void * _Nonnull self, float * _Nonnull * _Nonnull trainingData, float * _Nullable * _Nullable testData, size_t tr1, size_t tr2, size_t * _Nullable ts1, size_t * _Nullable ts2, int * _Nonnull ntLayers, size_t numberOfLayers, int * _Nonnull inoutSizes, int * _Nullable classifications, int epochs, int miniBatchSize, float eta, float lambda, bool * _Nullable showTotalCost);
    void (* _Nullable miniBatch)(void * _Nonnull self, float * _Nonnull * _Nonnull miniBatch, int miniBatchSize, int * _Nonnull ntLayers, size_t numberOfLayers, size_t tr1, float eta, float lambda);
    void(* _Nullable updateWeightsBiases)(void * _Nonnull self, int miniBatchSize, size_t tr1, float eta, float lambda);
    void (* _Nullable batchAccumulation)(void * _Nonnull self);
    void * _Nullable (* _Nullable backpropagation)(void * _Nonnull self);
    void (* _Nonnull feedforward)(void * _Nonnull self);
    int (* _Nullable evaluate)(void * _Nonnull self, float * _Nonnull * _Nonnull testData, size_t ts1);
    float (* _Nullable totalCost)(void * _Nonnull self, float * _Nonnull * _Nonnull data, size_t m, int * _Nullable classifications, float lambda, bool convert);
} NeuralNetwork;

NeuralNetwork * _Nonnull allocateNeuralNetwork(void);
