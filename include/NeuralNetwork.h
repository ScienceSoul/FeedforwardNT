//
//  NeuralNetwork.h
//  FeedforwardNT
//
//  Created by Seddik hakime on 31/05/2017.
//

#ifndef NeuralNetwork_h
#define NeuralNetwork_h

#include <stdio.h>
#include "Utils.h"
#include "TimeProfile.h"
#include "MetalCompute.h"

#endif /* NeuralNetwork_h */

typedef struct weightMatrixDimension {
    size_t m, n;
} weightMatrixDimension;

typedef struct biasVectorDimension {
    size_t n;
} biasVectorDimension;

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

typedef struct training {
    float * _Nullable *_Nullable set;
    size_t m, n;
    float * _Nullable * _Nullable (* _Nullable reader)(const char * _Nonnull fileName, size_t * _Nonnull len1, size_t * _Nonnull len2);
} training;
typedef struct test {
    float * _Nullable *_Nullable set;
    size_t m, n;
    float * _Nullable * _Nullable (* _Nullable reader)(const char * _Nonnull fileName, size_t * _Nonnull len1, size_t * _Nonnull len2);
} test;
typedef struct validation {
    float * _Nullable *_Nullable set;
    size_t m, n;
} validation;

typedef struct data {
    training * _Nullable training;
    test * _Nullable test;
    validation *_Nullable validation;
    void (* _Nullable init)(void * _Nonnull self);
    void (* _Nullable load)(void * _Nonnull self, const char * _Nonnull dataSetName, const char * _Nonnull trainFile, const char * _Nonnull testFile, bool testData);
} data;

typedef struct parameters {
    int epochs, miniBatchSize;
    size_t numberOfLayers, numberOfDataDivisions, numberOfClassifications, numberOfInouts;
    int ntLayers[100], dataDivisions[2], classifications[100], inoutSizes[2];
    float eta, lambda;
} parameters;

typedef struct NeuralNetwork {
    
    data * _Nullable data;
    float * _Nullable * _Nullable batch;
    parameters * _Nullable parameters;
    int (* _Nullable load)(void * _Nonnull self, const char * _Nonnull paraFile, char * _Nonnull dataSetName, char * _Nonnull dataSetFile);
    
    int example_idx;
    size_t number_of_features;
    size_t number_of_layers;
    size_t max_number_of_nodes_in_layer;
    
    float * _Nullable weights;
    float * _Nullable biases;
    weightMatrixDimension weightsDimensions[100];
    biasVectorDimension biasesDimensions[100];
    
    activationNode * _Nullable activationsList;
    zNode * _Nullable zsList;
    dcdwNode * _Nullable dcdwsList;
    dcdbNode * _Nullable dcdbsList;
    dcdwNode * _Nullable delta_dcdwsList;
    dcdbNode * _Nullable delta_dcdbsList;
    
    MetalCompute * _Nullable gpu;
        
    void (* _Nullable genesis)(void * _Nonnull self);
    void (* _Nullable finale)(void * _Nonnull self);
    void (* _Nullable gpu_alloc)(void * _Nonnull self);
    
    void (* _Nullable compute)(void * _Nonnull self, bool * _Nullable showTotalCost);
    void (* _Nullable miniBatch)(void * _Nonnull self, float * _Nonnull * _Nonnull miniBatch);
    void (* _Nullable updateWeightsBiases)(void * _Nonnull self);
    void (* _Nullable batchAccumulation)(void * _Nonnull self);
    void * _Nullable (* _Nullable backpropagation)(void * _Nonnull self);
    void (* _Nonnull feedforward)(void * _Nonnull self);
    void (* _Nonnull gpuFeedforward)(void * _Nonnull self);
    int (* _Nullable evaluate)(void * _Nonnull self);
    float (* _Nullable totalCost)(void * _Nonnull self, float * _Nonnull * _Nonnull data, size_t m, bool convert);
} NeuralNetwork;

NeuralNetwork * _Nonnull newNeuralNetwork(void);
