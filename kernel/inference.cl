inline  float sigmoid(float z);
inline float nanToNum(float value);

inline  float sigmoid(float z) {
    return 1.0f / (1.0f + exp(-z));
}

inline float nanToNum(float value) {
    float val = value;
    if (isnan(val) != 0) val = 0.0f;
    
    if (isinf(val) != 0) {
        if (val > 0) {
            val = HUGE_VALF;
        } else if (val < 0) {
            val= -HUGE_VALF;
        }
    }
    return val;
}

__kernel void inference(int M, int N, __global float *W, __global float *A, __global float *B, __global float *Z) {
    
    float sum, val=0.0f;
    
    int globalID = get_global_id(0);
    int start = globalID * N;
    W += start;
    
    sum = 0.0f;
    for(int j=0; j<N; j++) {
        sum += W[j] * A[j];
    }
    val = sum + B[globalID];
    val = sigmoid(val);
    val = nanToNum(val);
    Z[globalID] = val;
}


