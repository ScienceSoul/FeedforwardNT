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

#define PROGRAM_NAME "FeedforwardNT"

void __attribute__((overloadable)) fatal(char head[_Nonnull]);
void __attribute__((overloadable)) fatal(char head[_Nonnull], char message[_Nonnull]);
void __attribute__((overloadable)) fatal(char head[_Nonnull], char message[_Nonnull], int n);
void __attribute__((overloadable)) fatal(char head[_Nonnull], char message[_Nonnull], double n);
void __attribute__((overloadable)) warning(char head[_Nonnull], char message[_Nonnull]);
void __attribute__((overloadable)) warning(char head[_Nonnull], char message[_Nonnull], int n);
void __attribute__((overloadable)) warning(char head[_Nonnull], char message[_Nonnull], double n);

void shuffle(float * _Nonnull * _Nonnull array, size_t len1, size_t len2);
void parseArgument(const char * _Nonnull argument, const char * _Nonnull argumentName, int  * _Nonnull result, size_t * _Nonnull numberOfItems);
float randn(float mu, float sigma);

int __attribute__((overloadable)) min_array(int * _Nonnull a, size_t num_elements);
float __attribute__((overloadable)) min_array(float * _Nonnull a, size_t num_elements);

int __attribute__((overloadable)) max_array(int * _Nonnull a, size_t num_elements);
float __attribute__((overloadable)) max_array(float * _Nonnull a, size_t num_elements);

int __attribute__((overloadable)) argmax(int * _Nonnull a, size_t num_elements);
int __attribute__((overloadable)) argmax(float * _Nonnull a, size_t num_elements);

float sigmoid(float z);
float sigmoidPrime(float z);

float crossEntropyCost(float * _Nonnull a, float * _Nonnull y, size_t n);

float __attribute__((overloadable)) frobeniusNorm(float * _Nonnull * _Nonnull mat, size_t m, size_t n);
float __attribute__((overloadable)) frobeniusNorm(float * _Nonnull mat, size_t n);

void  __attribute__((overloadable)) nanToNum(float * _Nonnull array, size_t n);

int nearestPower2(int num);
