//
//  Utils.c
//  FeedforwardNT
//
//  Created by Seddik hakime on 21/05/2017.
//

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#endif

#include "Utils.h"
#include "Memory.h"

static int formatType;
void format(char * _Nullable head, char * _Nullable message, int *iValue, double *dValue);

void __attribute__((overloadable)) fatal(char head[]) {
    
    formatType = 1;
    format(head, NULL, NULL, NULL);
}

void __attribute__((overloadable)) fatal(char head[], char message[]) {
    
    formatType = 2;
    format(head, message, NULL, NULL);
}

void __attribute__((overloadable)) fatal(char head[], char message[], int n) {
    
    formatType = 3;
    format(head, message, &n, NULL);
}

void __attribute__((overloadable)) fatal(char head[], char message[], double n) {
    
    formatType = 4;
    format(head, message, NULL, &n);
}

void __attribute__((overloadable)) warning(char head[], char message[])
{
    fprintf(stdout, "%s: %s\n", head, message);
}

void __attribute__((overloadable)) warning(char head[], char message[], int n)
{
    fprintf(stdout, "%s: %s %d\n", head, message, n);
}

void __attribute__((overloadable)) warning(char head[], char message[], double n)
{
    fprintf(stdout, "%s: %s %f\n", head, message, n);
}

void format(char * _Nullable head, char * _Nullable message, int *iValue, double *dValue) {
    
    fprintf(stderr, "##                    A FATAL ERROR occured                   ##\n");
    fprintf(stderr, "##        Please look at the error log for diagnostic         ##\n");
    fprintf(stderr, "\n");
    if (formatType == 1) {
        fprintf(stderr, "%s: Program will abort...\n", head);
    } else if (formatType == 2) {
        fprintf(stderr, "%s: %s\n", head, message);
    } else if (formatType == 3) {
        fprintf(stderr, "%s: %s %d\n", head, message, *iValue);
    } else if (formatType == 4) {
        fprintf(stderr, "%s: %s %f\n", head, message, *dValue);
    }
    if (formatType == 2 || formatType == 3 || formatType == 4)
        fprintf(stderr, "Program will abort...\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "################################################################\n");
    fprintf(stderr, "################################################################\n");
    exit(-1);
}

void shuffle(float * _Nonnull * _Nonnull array, size_t len1, size_t len2) {
    
    float t[len2];
    
    if (len1 > 1)
    {
        for (int i = 0; i < len1 - 1; i++)
        {
            int j = i + rand() / (RAND_MAX / (len1 - i) + 1);
            for (int k=0; k<len2; k++) {
                t[k] = array[j][k];
            }
            for (int k=0; k<len2; k++) {
                array[j][k] = array[i][k];
            }
            for (int k=0; k<len2; k++) {
                array[i][k] = t[k];
            }
        }
    }
}

void parseArgument(const char * _Nonnull argument, const char * _Nonnull argumentName, int  * _Nonnull result, size_t * _Nonnull numberOfItems) {
    int idx = 0;
    *numberOfItems = 0;
    
    fprintf(stdout, "%s: parsing the parameter %s: %s.\n", PROGRAM_NAME, argumentName, argument);
    
    size_t len = strlen(argument);
    if (argument[0] != '{' || argument[len-1] != '}') fatal(PROGRAM_NAME, "input argument for network definition should start with <{> and end with <}>.");
    
    while (argument[idx] != '}') {
        if (argument[idx] == '{') {
            if (argument[idx+1] == ',' || argument[idx+1] == '{') fatal(PROGRAM_NAME, "syntax error <{,> or <{{> in imput argument for network definition.");
            idx++;
            continue;
        }
        if (argument[idx] == ',') {
            if (argument[idx+1] == '}' || argument[idx+1] == ',') fatal(PROGRAM_NAME, "syntax error <,}> or <,,> in imput argument for network definition.");
            (*numberOfItems)++;
            idx++;
            continue;
        } else {
            int digit = argument[idx] - '0';
            result[*numberOfItems] = result[*numberOfItems] * 10 + digit;
            idx++;
        }
    }
    (*numberOfItems)++;
}

// Generate random numbers from Normal Distribution (Gauss Distribution) with mean mu and standard deviation sigma
// using the Marsaglia and Bray method
float randn(float mu, float sigma) {
    
    float U1, U2, W, mult;
    static float X1, X2;
    static int call = 0;
    
    if (call == 1)
    {
        call = !call;
        return (mu + sigma * (float) X2);
    }
    
    do
    {
        U1 = -1 + ((float) rand () / RAND_MAX) * 2;
        U2 = -1 + ((float) rand () / RAND_MAX) * 2;
        W = pow (U1, 2) + pow (U2, 2);
    }
    while (W >= 1 || W == 0);
    
    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;
    
    call = !call;
    
    return (mu + sigma * (float) X1);
}

int __attribute__((overloadable)) min_array(int * _Nonnull a, size_t num_elements) {
    
    int min = INT_MAX;
    for (int i=0; i<num_elements; i++) {
        if (a[i] < min) {
            min = a[i];
        }
    }
    
    return min;
}

float __attribute__((overloadable)) min_array(float * _Nonnull a, size_t num_elements) {
    
    float min = INT_MAX;
    for (int i=0; i<num_elements; i++) {
        if (a[i] < min) {
            min = a[i];
        }
    }
    
    return min;
}

int __attribute__((overloadable)) max_array(int * _Nonnull a, size_t num_elements)
{
    int max = -INT_MAX;
    for (int i=0; i<num_elements; i++) {
        if (a[i] > max) {
            max = a[i];
        }
    }
    
    return max;
}

float __attribute__((overloadable)) max_array(float * _Nonnull a, size_t num_elements) {
    
    float max = -INT_MAX;
    for (int i=0; i<num_elements; i++) {
        if (a[i] > max) {
            max = a[i];
        }
    }
    
    return max;
}

int __attribute__((overloadable)) argmax(int * _Nonnull a, size_t num_elements) {
    
    int idx=0, max = -INT_MAX;
    for (int i=0; i<num_elements; i++) {
        if (a[i] > max) {
            max = a[i];
            idx = i;
        }
    }
    
    return idx;
}

int __attribute__((overloadable)) argmax(float * _Nonnull a, size_t num_elements) {
    
    int idx=0;
    float max = -HUGE_VAL;
    for (int i=0; i<num_elements; i++) {
        if (a[i] > max) {
            max = a[i];
            idx = i;
        }
    }
    
    return idx;
}

//  The sigmoid fonction
float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

// Derivative of the sigmoid function
float sigmoidPrime(float z) {
    return sigmoid(z) * (1.0f - sigmoid(z));
}

//
//  Compute the Frobenius norm of a m x n matrix
//
float frobeniusNorm(float * _Nonnull * _Nonnull mat, size_t m, size_t n) {
    
    float norm = 0.0f;
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            norm = norm + powf(mat[i][j], 2.0f);
        }
    }
    
    return sqrtf(norm);
}

float crossEntropyCost(float * _Nonnull a, float * _Nonnull y, size_t n) {
    
    float cost = 0.0f;
    float buffer[n];
    
    for (int i=0; i<n; i++) {
        buffer[i] = -y[i]*logf(a[i]) - (1.0f-y[i])*logf(1.0-a[i]);
    }
    nanToNum(buffer, n);
#ifdef __APPLE__
    vDSP_sve(buffer, 1, &cost, n);
#else
    for (int i=0; i<n; i++) {
        cost = cost + buffer[i];
    }
#endif
    
    return cost;
}

void  __attribute__((overloadable)) nanToNum(float * _Nonnull array, size_t n) {
    
    for (int i=0; i<n; i++) {
        if (isnan(array[i]) != 0) array[i] = 0.0f;
        
        if (isinf(array[i]) != 0) {
            if (array[i] > 0) {
                array[i] = HUGE_VALF;
            } else if (array[i] < 0) {
                array[i] = -HUGE_VALF;
            }
        }
    }
}
