//
//  Utils.h
//  FeedforwardNT
//
//  Created by Seddik hakime on 21/05/2017.
//

#ifndef Utils_h
#define Utils_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>

#include "Memory.h"

#endif /* Utils_h */

void __attribute__((overloadable)) fatal(char head[_Nonnull]);
void __attribute__((overloadable)) fatal(char head[_Nonnull], char message[_Nonnull]);
void __attribute__((overloadable)) fatal(char head[_Nonnull], char message[_Nonnull], int n);
void __attribute__((overloadable)) fatal(char head[_Nonnull], char message[_Nonnull], double n);
void __attribute__((overloadable)) warning(char head[_Nonnull], char message[_Nonnull]);
void __attribute__((overloadable)) warning(char head[_Nonnull], char message[_Nonnull], int n);
void __attribute__((overloadable)) warning(char head[_Nonnull], char message[_Nonnull], double n);

int loadParameters(const char * _Nonnull paraFile, char * _Nonnull dataSetName, char * _Nonnull dataSetFile, int * _Nonnull ntLayers, size_t * _Nonnull numberOfLayers, int * _Nonnull dataDivisions, size_t * _Nonnull numberOfDataDivisions, int * _Nonnull classifications, size_t * _Nonnull numberOfClassifications, int * _Nonnull inoutSizes, size_t * _Nonnull numberOfInouts, int * _Nonnull epochs, int * _Nonnull miniBatchSize, float * _Nonnull eta, float * _Nonnull lambda);

float * _Nonnull * _Nonnull loadData(const char * _Nonnull dataSetName, const char * _Nonnull fileName, size_t * _Nonnull len1, size_t * _Nonnull len2);

float * _Nonnull * _Nonnull createTrainigData(float * _Nonnull * _Nonnull dataSet, size_t start, size_t end, size_t * _Nonnull t1, size_t * _Nonnull t2, int * _Nonnull classifications, size_t numberOfClassifications, int * _Nonnull inoutSizes);

float * _Nonnull * _Nonnull createTestData(float * _Nonnull * _Nonnull dataSet, size_t len1, size_t len2, size_t start, size_t end, size_t * _Nonnull t1, size_t * _Nonnull t2);

void shuffle(float * _Nonnull * _Nonnull array, size_t len1, size_t len2);
void parseArgument(const char * _Nonnull argument, const char * _Nonnull argumentName, int * _Nonnull result, size_t * _Nonnull numberOfItems);
float randn(float mu, float sigma);

int __attribute__((overloadable)) min_array(int * _Nonnull a, size_t num_elements);
int __attribute__((overloadable)) max_array(int * _Nonnull a, size_t num_elements);

int __attribute__((overloadable)) argmax(int * _Nonnull a, size_t num_elements);
int __attribute__((overloadable)) argmax(float * _Nonnull a, size_t num_elements);

float sigmoid(float z);
float sigmoidPrime(float z);

float crossEntropyCost(float * _Nonnull a, float * _Nonnull y, size_t n);

float frobeniusNorm(float * _Nonnull * _Nonnull mat, size_t m, size_t n);

void  __attribute__((overloadable)) nanToNum(float * _Nonnull array, size_t n);
