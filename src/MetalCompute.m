//
//  MetalCompute.m
//  FeedforwardNT
//
//  Created by Hakime Seddik on 08/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <sys/stat.h>

#include "MetalCompute.h"
#include "Utils.h"
#include "TimeProfile.h"

id <MTLDevice> _Nullable device;
id <MTLCommandQueue> _Nullable commandQueue;
id <MTLLibrary> _Nullable library;

NSMutableArray *functions;
NSMutableArray *pipelineStates;

id <MTLBuffer> firstInputBuffer;
id <MTLBuffer> secondInputBuffer;
id <MTLBuffer> dimBuffer_uint16;
id <MTLBuffer> dimBuffer_uint32;
id <MTLBuffer> matrixBuffer;
id <MTLBuffer> vectorBuffer;
id <MTLBuffer> outputBuffer;

size_t buffer_size;
bool allocation;

int LoadFileIntoString(const char * _Nonnull filename, char * _Nonnull * _Nullable text, size_t * _Nonnull len) {
    struct stat statbuf;
    FILE        *fh;
    int         file_len;
    
    fh = fopen(filename, "r");
    if (fh == 0)
        return -1;
    
    stat(filename, &statbuf);
    file_len = (int)statbuf.st_size;
    *len = file_len;
    *text = (char *) malloc(file_len + 1);
    fread(*text, file_len, 1, fh);
    (*text)[file_len] = '\0';
    
    fclose(fh);
    return 0;
}

void init (void) {
    device = MTLCreateSystemDefaultDevice();
    commandQueue = device.newCommandQueue;
    
    char * metal_source;
    size_t src_len;
    
    if (LoadFileIntoString("metal/MetalKernels.metal", &metal_source, &src_len) != 0) {
        if (LoadFileIntoString("../metal/MetalKernels.metal", &metal_source, &src_len) != 0) {
            fatal(PROGRAM_NAME, "<metal compute>: can't load the metal source file.");
        }
    }
    
    NSError *error;
    library = [device newLibraryWithSource:[NSString stringWithUTF8String:metal_source] options:NULL error:&error];
    if (error != nil) {
        fprintf(stderr, "<metal compute>: error when creating a new library state.");
        fprintf(stderr, "<metal compute>: error code: %ld\n", (long)error.code);
        fatal(PROGRAM_NAME, "Program will abort.");
    }
    
    functions = [NSMutableArray new];
    pipelineStates = [NSMutableArray new];
    
    buffer_size = 0;
    allocation = false;
}

void nullify(void) {
    device = nil;
    commandQueue = nil;
    library = nil;
    functions = nil;
    pipelineStates = nil;
    
    firstInputBuffer = nil;
    secondInputBuffer = nil;
    dimBuffer_uint16 = nil;
    dimBuffer_uint32 = nil;
    matrixBuffer = nil;
    vectorBuffer = nil;
    outputBuffer = nil;
}

void allocate_buffers(size_t n) {
    if (!allocation) {
        buffer_size = n;
        firstInputBuffer = [device newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
        secondInputBuffer = [device newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
        dimBuffer_uint16 = [device newBufferWithLength:sizeof(UInt16) options:(MTLResourceStorageModeShared)];
        dimBuffer_uint32 = [device newBufferWithLength:2*sizeof(UInt32) options:MTLResourceStorageModeShared];
        matrixBuffer = [device newBufferWithLength:(n*n)*sizeof(float) options:MTLResourceStorageModeShared];
        vectorBuffer = [device newBufferWithLength:n options:MTLResourceStorageModeShared];
        outputBuffer = [device newBufferWithLength:n options:MTLResourceStorageModeShared];
        
        void *buffer = firstInputBuffer.contents;
        memset(buffer, 0.0f, buffer_size*sizeof(float));
        
        buffer = secondInputBuffer.contents;
        memset(buffer, 0.0f, buffer_size*sizeof(float));
        
        buffer = dimBuffer_uint16.contents;
        memset(buffer, 0, sizeof(UInt16));
        
        buffer = dimBuffer_uint32.contents;
        memset(buffer, 0, sizeof(UInt32));
        
        buffer = matrixBuffer.contents;
        memset(buffer, 0.0f, (buffer_size*buffer_size)*sizeof(float));
        
        buffer = vectorBuffer.contents;
        memset(buffer, 0.0f, buffer_size*sizeof(float));
        
        buffer = outputBuffer.contents;
        memset(buffer, 0.0f, buffer_size*sizeof(float));
    }
}

void prepare (char * _Nonnull task) {
    
    id <MTLFunction> function;
    id <MTLComputePipelineState> pipelineState;
    
    if (!allocation) {
        if (strcmp(task, "feedforward") == 0) {
            function = [library newFunctionWithName:[NSString stringWithUTF8String:"activation_sigmoid"]];
            [functions addObject:function];
            
            NSError *error;
            pipelineState = [device newComputePipelineStateWithFunction:functions[0] error:&error];
            if (error != nil) {
                fprintf(stderr, "<metal compute>: error when creating a pipeline state.");
                fprintf(stderr, "<metal compute>: error code: %ld\n", (long)error.code);
                fatal(PROGRAM_NAME, "Program will abort.");
            }
            
            [functions addObject:function];
            [pipelineStates addObject:pipelineState];
        }
        allocation = true;
    }
}

void compute_activation(float * _Nonnull wa, float * _Nonnull b, float * _Nonnull a, size_t n) {
    
    MTLSize theadGroupSize;
    MTLSize threadGroups;
    
    void *buff = firstInputBuffer.contents;
    memcpy(buff, wa, n*sizeof(float));
    
    buff = secondInputBuffer.contents;
    memcpy(buff, b, n*sizeof(float));
    
    theadGroupSize = MTLSizeMake(128, 1, 1);
    threadGroups = MTLSizeMake(0, 1, 1);
    
    int pow = nearestPower2((int)n);
    threadGroups.width = pow;
    
    @autoreleasepool{
        id <MTLCommandBuffer> commandBuffer = commandQueue.commandBuffer;
        id <MTLComputeCommandEncoder> commandEncoder = commandBuffer.computeCommandEncoder;
        [commandEncoder setComputePipelineState:pipelineStates[0]];
        
        [commandEncoder setBuffer:firstInputBuffer offset:0 atIndex:0];
        [commandEncoder setBuffer:secondInputBuffer offset:0 atIndex:1];
        [commandEncoder setBuffer:outputBuffer offset:0 atIndex:2];
        
        [commandEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:theadGroupSize];
        
        [commandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    void *data = outputBuffer.contents;
    memcpy(a, data, n*sizeof(float));
}

MetalCompute * _Nonnull metalCompute(void) {
    
    MetalCompute *metalComppute = (MetalCompute *)malloc(sizeof(MetalCompute));
    
    metalComppute->init = init;
    metalComppute->prepare = prepare;
    metalComppute->allocate_buffers = allocate_buffers;
    metalComppute->nullify = nullify;
    
    metalComppute->activation = compute_activation;
    
    return metalComppute;
}

