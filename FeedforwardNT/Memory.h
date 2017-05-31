//
//  memory.h
//  FeedforwardNT
//
//  Created by Seddik hakime on 21/05/2017.
//

#ifndef memory_h
#define memory_h

#include <stddef.h>
#include <stdarg.h>

#include "Utils.h"

#endif /* Memory_h */

int * __nonnull intvec(long nl, long nh);
float * __nonnull floatvec(long nl, long nh);
float * __nonnull * __nonnull floatmatrix(long nrl, long nrh, long ncl, long nch);
void free_ivector(int * __nonnull v, long nl, long nh);
void free_fvector(float * __nonnull v, long nl, long nh);
void free_fmatrix(float * __nonnull * __nonnull m, long nrl, long nrh, long ncl, long nch);
