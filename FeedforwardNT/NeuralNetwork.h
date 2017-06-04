//
//  NeuralNetwork.h
//  FeedforwardNT
//
//  Created by Seddik hakime on 31/05/2017.
//

#ifndef NeuralNetwork_h
#define NeuralNetwork_h

#include <stdio.h>
#include <pthread.h>
#include "Utils.h"
#include "TimeProfile.h"

#endif /* NeuralNetwork_h */

typedef struct weightNode {
    size_t m, n;
    float * __nullable * __nullable w;
    struct weightNode * __nullable next;
    struct weightNode * __nullable previous;
} weightNode;

typedef struct biasNode {
    size_t n;
    float * __nullable b;
    struct biasNode * __nullable next;
    struct biasNode * __nullable previous;
} biasNode;

typedef struct activationNode {
    size_t n;
    float * __nullable a;
    struct activationNode * __nullable next;
    struct activationNode * __nullable previous;
} activationNode;

typedef struct zNode {
    size_t n;
    float * __nullable z;
    struct zNode * __nullable next;
    struct zNode * __nullable previous;
} zNode;

typedef struct dcdwNode {
    size_t m, n;
    float * __nullable * __nullable dcdw;
    struct dcdwNode * __nullable next;
    struct dcdwNode * __nullable previous;
} dcdwNode;

typedef struct dcdbNode {
    size_t n;
    float * __nullable dcdb;
    struct dcdbNode * __nullable next;
    struct dcdbNode * __nullable previous;
} dcdbNode;

typedef struct pthreadBatchNode {
    int index;
    int max;
    float * __nullable * __nullable batch;
    struct weightNode * __nullable weightsList;
    struct biasNode * __nullable biasesList;
    struct activationNode * __nullable activationsList;
    struct zNode * __nullable zsList;
    struct dcdwNode * __nullable dcdwsList;
    struct dcdbNode * __nullable dcdbsList;
    int * __nonnull inoutSizes;
} pthreadBatchNode;

typedef struct NeuralNetwork {
    weightNode * __nullable weightsList;
    biasNode * __nullable biasesList;
    activationNode * __nullable activationsList;
    zNode * __nullable zsList;
    dcdwNode * __nullable dcdwsList;
    dcdbNode * __nullable dcdbsList;
    
    pthreadBatchNode * __nullable * __nullable threadDataPt;
    pthread_t __nullable * __nullable threadTID;
    
    void (* __nullable create)(void * __nonnull self, int * __nonnull ntLayers, size_t numberOfLayers, int * __nullable miniBatchSize, bool pthread);
    void (* __nullable destroy)(void * __nonnull self, int * __nullable miniBatchSize, bool pthread);
    
    void (* __nullable SDG)(void * __nonnull self, float * __nonnull * __nonnull trainingData, float * __nullable * __nullable testData, size_t tr1, size_t tr2, size_t * __nullable ts1, size_t * __nullable ts2, int * __nonnull ntLayers, size_t numberOfLayers, int * __nonnull inoutSizes, int * __nullable classifications, int epochs, int miniBatchSize, float eta, float lambda, bool pthread, bool * __nullable showTotalCost);
    void (* __nullable updateMiniBatch)(void * __nonnull self, float * __nonnull * __nonnull miniBatch, int miniBatchSize, int * __nonnull ntLayers, size_t numberOfLayers, size_t tr1, float eta, float lambda, bool * __nullable pthread);
    void(* __nullable updateWeightsBiases)(void * __nonnull self, int miniBatchSize, size_t tr1, float eta, float lambda);
    void (* __nullable accumulateFromThreads)(void * __nonnull self, int miniBatchSize, bool pthread);
    void * __nullable (* __nullable backpropagation)(void * __nonnull node);
    void (* __nonnull feedforward)(void * __nonnull self);
    int (* __nullable evaluate)(void * __nonnull self, float * __nonnull * __nonnull testData, size_t ts1, int * __nonnull inoutSizes);
    float (* __nullable totalCost)(void * __nonnull self, float * __nonnull * __nonnull data, size_t m, int * __nonnull inoutSizes, int * __nullable classifications, float lambda, bool convert);
} NeuralNetwork;

NeuralNetwork * __nonnull allocateNeuralNetwork(void);
