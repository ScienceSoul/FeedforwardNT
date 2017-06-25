//
//  LoadIrisDataSet.c
//  FeedforwardNT
//
//  Created by Seddik hakime on 21/05/2017.
//

#include "Memory.h"
#include "LoadIrisDataSet.h"

float **readFile(const char * _Nonnull file, size_t * _Nonnull len) {
    
    FILE *f1 = fopen(file,"r");
    if(!f1) {
        return NULL;
    }
    
    // Figure out the number of inputs
    int inputs = 0;
    char ch = 0;
    do {
        ch = fgetc(f1);
        if(ch == '\n'){
            inputs++;
        }
    }
    while (!feof(f1));
    fprintf(stdout, "FeedforwardNT: number of inputs in iris data set: %d.\n", inputs);
    rewind(f1);

    // Allocate some buffers
    *len = inputs;
    float *data1 = floatvec(0, *len-1);
    float *data2 = floatvec(0, *len-1);
    float *data3 = floatvec(0, *len-1);
    float *data4 = floatvec(0, *len-1);
    int *data5 = intvec(0, *len-1);
    if (data1 == NULL || data2 == NULL || data3 == NULL || data4 == NULL || data5 == NULL) {
        fatal("FeedforwardNT", "allocation error in readFile.");
    }
    memset(data1, 0.0f, *len*sizeof(float));
    memset(data2, 0.0f, *len*sizeof(float));
    memset(data3, 0.0f, *len*sizeof(float));
    memset(data4, 0.0f, *len*sizeof(float));
    memset(data5, 0, *len*sizeof(int));
    
    inputs=0;
    char class[256];
    do {
        fscanf(f1,"%f,%f,%f,%f,%s\n", &data1[inputs], &data2[inputs], &data3[inputs], &data4[inputs], class);
        // We don't really need to store the iris class strings as is, just assign to them a classification number
        // 0: Iris-setosa
        // 1: Iris-versicolor
        // 2: Iris-virginica
        if (strcmp(class, "Iris-setosa") == 0) {
            data5[inputs] = 0.0f;
        } else if (strcmp(class, "Iris-versicolor") == 0) {
            data5[inputs] = 1.0f;
        } else {
            data5[inputs] = 2.0f;
        }
        inputs++;
    }
    while (!feof(f1));
    fclose(f1);
    
    // Return a matrix of the data set
    float **dataSet = floatmatrix(0, *len-1, 0, 5-1);
    memset(*dataSet, 0.0f, (*len*5)*sizeof(float));
    for (int i=0; i<*len; i++) {
        dataSet[i][0] = data1[i];
        dataSet[i][1] = data2[i];
        dataSet[i][2] = data3[i];
        dataSet[i][3] = data4[i];
        dataSet[i][4] = (float)data5[i];
    }
    
    free_fvector(data1, 0, *len-1);
    free_fvector(data2, 0, *len-1);
    free_fvector(data3, 0, *len-1);
    free_fvector(data4, 0, *len-1);
    free_ivector(data5, 0, *len-1);
    return dataSet;
}

float **loadIris(const char * _Nonnull file, size_t * _Nonnull len) {
    
    float **dataSet = readFile(file, len);
    if (dataSet == NULL) {
        fatal("FeedforwardNT", "problem reading iris data set.");
    } else fprintf(stdout, "FeedforwardNT: load iris data set.\n");
    
    return dataSet;
}
