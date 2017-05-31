//
//  Training.h
//  FeedforwardNT
//
//  Created by Seddik hakime on 21/05/2017.
//

#ifndef Training_h
#define Training_h

#include <pthread.h>
#include "Utils.h"

#endif /* Training_h */

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

void SDG(float * __nonnull * __nonnull trainingData, float * __nonnull * __nonnull testData, size_t tr1, size_t tr2, size_t ts1, size_t ts2, int * __nonnull ntLayers, size_t numberOfLayers, int * __nonnull inoutSizes, int * __nonnull classifications, int epochs, int miniBatchSize, float eta, float lambda, bool pthread);

void updateMiniBatch(float * __nonnull * __nonnull miniBatch,
                     int miniBatchSize,
                     int * __nonnull inoutSizes,
                     weightNode * __nonnull weightsList,
                     biasNode * __nonnull biasesList,
                     dcdwNode * __nullable dcdwsList,
                     dcdbNode * __nullable dcdbsList,
                     pthreadBatchNode * __nullable * __nullable threadDataPt,
                     pthread_t __nullable * __nullable  threadTID,
                     int * __nonnull ntLayers,
                     size_t numberOfLayers, size_t tr1, float eta, float lambda, bool * __nullable pthread);

void updateWeightsBiases(weightNode * __nonnull weightsList, biasNode * __nonnull biasesList, dcdwNode * __nonnull dcdwsList, dcdbNode * __nonnull dcdbsList, int miniBatchSize, size_t tr1, float eta, float lambda);

void accumulateFromThreads(dcdwNode * __nonnull dcdwsList, dcdbNode * __nonnull dcdbsList, pthreadBatchNode * __nullable * __nullable threadDataPt, int miniBatchSize, bool pthread);

void * __nullable backpropagation(void * __nonnull node);

weightNode * __nonnull allocateWeightNode(void);
biasNode * __nonnull allocateBiasNode(void);

activationNode * __nonnull allocateActivationNode(void);
zNode * __nonnull allocateZNode(void);

dcdwNode * __nonnull allocateDcdwNode(void);
dcdbNode * __nonnull allocateDcdbNode(void);

weightNode * __nonnull initWeightsList(int * __nonnull ntLayers, size_t numberOfLayers);
biasNode * __nonnull initBiasesList(int * __nonnull ntLayers, size_t numberOfLayers);

activationNode * __nonnull initActivationsList(int * __nonnull ntLayers, size_t numberOfLayers);
zNode * __nonnull initZsList(int * __nonnull ntLayers, size_t numberOfLayers);

dcdwNode * __nonnull initDcdwList(int * __nonnull ntLayers, size_t numberOfLayers);
dcdbNode * __nonnull initDcdbList(int * __nonnull ntLayers, size_t numberOfLayers);

pthreadBatchNode * __nonnull allocatePthreadBatchNode(void);

void deallocate(weightNode * __nonnull weightsList, biasNode * __nonnull biasesList, dcdwNode * __nonnull dcdwsList, dcdbNode * __nonnull dcdbsList, activationNode * __nonnull activationsList, zNode * __nonnull zsList, pthreadBatchNode * __nullable * __nullable threadDataPt, pthread_t __nullable * __nullable  threadTID, int miniBatchSize, bool pthread);

float sigmoid(float z);
float sigmoidPrime(float z);

void feedforward(weightNode * __nonnull weights, activationNode * __nonnull activations, biasNode * __nonnull biases, zNode * __nonnull zs);

int evaluate(float * __nonnull * __nonnull testData, size_t ts1, int * __nonnull inoutSizes, weightNode * __nonnull weightsList, biasNode * __nonnull biasesList, activationNode * __nonnull activationsList, zNode * __nonnull zsList);

float totalCost(float * __nonnull * __nonnull data, size_t m, weightNode * __nonnull weightsList, biasNode * __nonnull biasesList, activationNode * __nonnull activationsList, zNode * __nonnull zsList, int * __nonnull inoutSizes, int * __nonnull classifications, float lambda, bool convert);

float crossEntropyCost(float * __nonnull a, float * __nonnull y, size_t n);

float frobeniusNorm(float * __nonnull * __nonnull mat, size_t m, size_t n);
