//
//  NeuralNetwork.h
//  FeedforwardNT
//
//  Created by Seddik hakime on 31/05/2017.
//

#ifndef NeuralNetwork_h
#define NeuralNetwork_h

#include "MetalCompute.h"
#include "Utils.h"

#define MAX_NUMBER_NETWORK_LAYERS 100

typedef struct weightMatrixDimension {
    unsigned int m, n;
} weightMatrixDimension;

typedef struct biasVectorDimension {
    unsigned int n;
} biasVectorDimension;

typedef struct activationNode {
    unsigned int n;
    float * _Nullable a;
    struct activationNode * _Nullable next;
    struct activationNode * _Nullable previous;
} activationNode;

typedef struct zNode {
    unsigned int n;
    float * _Nullable z;
    struct zNode * _Nullable next;
    struct zNode * _Nullable previous;
} zNode;

typedef struct dcdwNode {
    unsigned int m, n;
    float * _Nullable * _Nullable dcdw;
    struct dcdwNode * _Nullable next;
    struct dcdwNode * _Nullable previous;
} dcdwNode;

typedef struct dcdbNode {
    unsigned int n;
    float * _Nullable dcdb;
    struct dcdbNode * _Nullable next;
    struct dcdbNode * _Nullable previous;
} dcdbNode;

typedef struct training {
    float * _Nullable *_Nullable set;
    unsigned int m, n;
    float * _Nullable * _Nullable (* _Nullable reader)(const char * _Nonnull fileName, unsigned int * _Nonnull len1, unsigned int * _Nonnull len2);
} training;
typedef struct test {
    float * _Nullable *_Nullable set;
    unsigned int m, n;
    float * _Nullable * _Nullable (* _Nullable reader)(const char * _Nonnull fileName, unsigned int * _Nonnull len1, unsigned int * _Nonnull len2);
} test;
typedef struct validation {
    float * _Nullable *_Nullable set;
    unsigned int m, n;
} validation;

typedef struct data {
    training * _Nullable training;
    test * _Nullable test;
    validation *_Nullable validation;
    void (* _Nullable init)(void * _Nonnull self);
    void (* _Nullable load)(void * _Nonnull self, const char * _Nonnull dataSetName, const char * _Nonnull trainFile, const char * _Nonnull testFile, bool testData);
} data;

typedef struct parameters {
    unsigned int number_of_suported_parameters;
    int epochs, miniBatchSize;
    unsigned int numberOfLayers, numberOfClassifications, numberOfActivationFunctions;
    float eta, lambda;
    
    char supported_parameters[10][256];
    char data[256], dataName[256];
    int topology[MAX_NUMBER_NETWORK_LAYERS], split[2], classifications[MAX_NUMBER_NETWORK_LAYERS];
    char activationFunctions[MAX_NUMBER_NETWORK_LAYERS][128];
} parameters;

typedef struct NeuralNetwork {
    
    data * _Nullable data;
    float * _Nullable * _Nullable batch;
    parameters * _Nullable parameters;
    int (* _Nullable load)(void * _Nonnull self, const char * _Nonnull paraFile);
    
    int example_idx;
    unsigned int number_of_parameters;
    unsigned int number_of_features;
    unsigned int number_of_layers;
    unsigned int max_number_of_nodes_in_layer;
    
    float * _Nullable weights;
    float * _Nullable biases;
    weightMatrixDimension weightsDimensions[MAX_NUMBER_NETWORK_LAYERS];
    biasVectorDimension biasesDimensions[MAX_NUMBER_NETWORK_LAYERS];
    
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
    float (* _Nonnull activationFunctions[MAX_NUMBER_NETWORK_LAYERS])(float z, float * _Nullable vec, unsigned int * _Nullable n);
    float (* _Nonnull activationDerivatives[MAX_NUMBER_NETWORK_LAYERS])(float z);
    int (* _Nullable evaluate)(void * _Nonnull self);
    float (* _Nullable totalCost)(void * _Nonnull self, float * _Nonnull * _Nonnull data, unsigned int m, bool convert);
} NeuralNetwork;

NeuralNetwork * _Nonnull newNeuralNetwork(void);

#endif /* NeuralNetwork_h */

