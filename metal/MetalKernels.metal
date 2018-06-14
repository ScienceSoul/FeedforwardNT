//
//  MetalKernels.metal
//  FeedforwardNT
//
//  Hakime Seddik on 11/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//
//  Created by Jorden Hill on 10/9/15.
//  Copyright © 2015 Jorden Hill. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

#define WARP_SIZE 32


kernel void activation_sigmoid(device float *x [[ buffer(0) ]],
                               device float *y [[ buffer(1) ]],
                               device float *z [[ buffer(2) ]],
                               uint id [[ thread_position_in_grid ]]) {
    y[id] = x[id] + y[id];
    z[id] =  1.0 / (1.0 + exp(-y[id]));
}

kernel void activation_tanh(device float *x [[ buffer(0) ]],
                            device float *y [[ buffer(1) ]],
                            device float *z [[ buffer(2) ]],
                            uint id [[ thread_position_in_grid ]]) {
    y[id] = x[id] + y[id];
    z[id] = tanh(y[id]);
}

kernel void activation_relu(device float *x [[ buffer(0) ]],
                            device float *y [[ buffer(1) ]],
                            device float *z [[ buffer(2) ]],
                            uint id [[ thread_position_in_grid ]]) {
    y[id] = x[id] + y[id];
    z[id] = fmax(y[id], 0.0);
}


kernel void sigmoid_prime(device float *x[[buffer(0)]],
                          device float *y [[buffer(1)]],
                          uint id [[thread_position_in_grid]])
{
    y[id] = (1.0 / (1.0 + exp(-x[id]))) * (1.0 - (1.0 / (1.0 + exp(-x[id]))));
}

kernel void tanh_prime(device float *x [[buffer(0)]],
                       device float *y [[buffer(1)]],
                       uint id [[thread_position_in_grid]])
{
    y[id] = 1 - pow(tanh(x[id]), 2);
}

kernel void relu_prime(device float *x [[buffer(0)]],
                       device float *y [[buffer(1)]],
                       uint id [[thread_position_in_grid]])
{
    y[id] = x[id] <= 0.0 ? 0.0 : 1.0;
}

