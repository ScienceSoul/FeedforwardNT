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
#include "LoadIrisDataSet.h"
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

int loadParameters(const char * _Nonnull paraFile, char * _Nonnull dataSetName, char * _Nonnull dataSetFile, int * _Nonnull ntLayers, size_t * _Nonnull numberOfLayers, int * _Nonnull dataDivisions, size_t * _Nonnull numberOfDataDivisions, int * _Nonnull classifications, size_t * _Nonnull numberOfClassifications, int * _Nonnull inoutSizes, size_t * _Nonnull numberOfInouts, int * _Nonnull epochs, int * _Nonnull miniBatchSize, float * _Nonnull eta, float * _Nonnull lambda) {
    
    // Very basic parsing of input parameters file.
    // TODO: Needs to change that to something more flexible and with better input validation
    
    FILE *f1 = fopen(paraFile,"r");
    if(!f1) {
        fprintf(stdout,"%s: can't open the input parameters file.\n", PROGRAM_NAME);
        return -1;
    }
    
    char string[256];
    int lineCount = 1;
    int empty = 0;
    while (1) {
        fscanf(f1,"%s\n", string);
        
        if (lineCount == 1 && string[0] != '{') {
            fatal(PROGRAM_NAME, "syntax error in the file for the input parameters.");
        } else if (lineCount == 1) {
            lineCount++;
            continue;
        } else if(string[0] == '\0') {
            empty++;
            if (empty > 1000) {
                fatal(PROGRAM_NAME, "syntax error in the file for the input keys. File should end with <}>.");
            }
            continue;
        }
        
        if (string[0] == '!') continue; // Comment line
        if (string[0] == '}') break;    // End of file
        
        if (lineCount == 2) {
            memcpy(dataSetName, string, strlen(string)*sizeof(char));
        }
        if (lineCount == 3) {
            memcpy(dataSetFile, string, strlen(string)*sizeof(char));
        }
        if (lineCount == 4) {
            parseArgument(string, "network definition", ntLayers, numberOfLayers);
        }
        if (lineCount == 5) {
            parseArgument(string, "data divisions", dataDivisions, numberOfDataDivisions);
        }
        if (lineCount == 6) {
            parseArgument(string, "classifications", classifications, numberOfClassifications);
        }
        if (lineCount == 7) {
            parseArgument(string, "inouts", inoutSizes, numberOfInouts);
        }
        if (lineCount == 8) {
            *epochs = atoi(string);
        }
        if (lineCount == 9) {
            *miniBatchSize = atoi(string);
        }
        if (lineCount == 10) {
            *eta = strtof(string, NULL);
        }
        if (lineCount == 11) {
            *lambda = strtof(string, NULL);
        }
        lineCount++;
    }
    
    if (*numberOfDataDivisions != 2) {
        fprintf(stdout,"%s: input data set should only be divided in two parts: one for training, one for testing.\n", PROGRAM_NAME);
        return -1;
    }
    if (*numberOfInouts != 2) {
        fprintf(stdout,"%s: only define one size for inputs and one size for outputs.\n", PROGRAM_NAME);
        return -1;
    }
    if (inoutSizes[1] < *numberOfClassifications || inoutSizes[1] > *numberOfClassifications) {
        fprintf(stdout,"%s: mismatch between number of classifications and the number of outputs.\n", PROGRAM_NAME);
        return -1;
    }
    
    return 0;
}

float **loadData(const char * _Nonnull dataSetName, const char * _Nonnull fileName, size_t * _Nonnull len1, size_t * _Nonnull len2) {
    
    float **dataSet = NULL;
    // Load data set
    // Right now this program basically only supports the Iris data set as input hence only
    // employs a very primitive way to parse an input data set file
    if (strcmp(dataSetName, "iris") == 0) {
        dataSet = loadIris(fileName, len1);
        *len2 = 5;
        // Shuffle the original data set
        shuffle(dataSet, *len1, *len2);
    } else {
        fatal(PROGRAM_NAME, "training the network without anything else than the iris data set is not yet supported.");
    }
    return dataSet;
}

float * _Nonnull * _Nonnull createTrainigData(float * _Nonnull * _Nonnull dataSet, size_t start, size_t end, size_t * _Nonnull t1, size_t * _Nonnull t2, int * _Nonnull classifications, size_t numberOfClassifications, int * _Nonnull inoutSizes) {
    
    int idx;
    float **trainingData = NULL;
    trainingData = floatmatrix(0, end-1, 0, (inoutSizes[0]+inoutSizes[1])-1);
    *t1 = end;
    *t2 = inoutSizes[0]+inoutSizes[1];
    
    if (inoutSizes[1] != numberOfClassifications) {
        fatal(PROGRAM_NAME, "the number of classifications should be equal to the number of activations.");
    }
    
    for (int i=0; i<end; i++) {
        for (int j=0; j<inoutSizes[0]; j++) {
            trainingData[i][j] = dataSet[i][j];
        }
        
        idx = inoutSizes[0];
        for (int k=0; k<inoutSizes[1]; k++) {
            trainingData[i][idx] = 0.0f;
            idx++;
        }
        for (int k=0; k<numberOfClassifications; k++) {
            if (dataSet[i][inoutSizes[0]] == classifications[k]) {
                trainingData[i][inoutSizes[0]+k] = 1.0f;
            }
        }
    }
    
    return trainingData;
}

float * _Nonnull * _Nonnull createTestData(float * _Nonnull * _Nonnull dataSet, size_t len1, size_t len2, size_t start, size_t end, size_t * _Nonnull t1, size_t * _Nonnull t2) {
    
    float **testData = floatmatrix(0, end, 0, len2-1);
    *t1 = end;
    *t2 = len2;
    
    int idx = 0;
    for (int i=(int)start; i<start+end; i++) {
        for (int j=0; j<len2; j++) {
            testData[idx][j] = dataSet[i][j];
        }
        idx++;
    }
    return testData;
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

void parseArgument(const char * _Nonnull argument, const char * _Nonnull argumentName, int * _Nonnull result, size_t * _Nonnull numberOfItems) {
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