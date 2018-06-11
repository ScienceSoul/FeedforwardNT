//
//  MetalCompute.m
//  FeedforwardNT
//
//  Created by Hakime Seddik on 08/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#import "MetalCompute.h"

id <MTLDevice> _Nullable device;
id <MTLCommandQueue> _Nullable commandQueue;
id <MTLComputePipelineState> _Nullable pipelineState;
id <MTLCommandBuffer> _Nullable commandBuffer;
id <MTLLibrary> _Nullable library;

void initCompute (void) {
    device = MTLCreateSystemDefaultDevice();
    commandQueue = device.newCommandQueue;
    pipelineState = NULL;
    library = device.newDefaultLibrary;
}

MetalCompute * _Nonnull metalCompute(void) {
    
    MetalCompute *metalComppute = (MetalCompute *)malloc(sizeof(MetalCompute));
    
    metalComppute->init = initCompute;
    return metalComppute;
}

