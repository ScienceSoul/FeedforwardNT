//
//  MetalCompute.h
//  FeedforwardNT
//
//  Created by Hakime Seddik on 08/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#ifndef MetalCompute_h
#define MetalCompute_h


typedef struct MetalCompute {
    
    void (* _Nullable init)(void);
    void (* _Nullable prepare)(char * _Nonnull task);
    void (* _Nullable allocate_buffers)(size_t n);
    void (* _Nullable nullify)(void);
    
    void (* _Nullable activation)(float * _Nonnull wa, float * _Nonnull b, float * _Nonnull a, size_t n);
    
} MetalCompute;

MetalCompute * _Nonnull metalCompute(void);

#endif /* MetalCompute_h */
