//
//  LoadMNISTDataSet.c
//  FeedforwardNT
//
//  Created by Hakime Seddik on 01/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#include <stdio.h>
#include "LoadMNISTDataSet.h"
#include "Utils.h"

void endianSwap(unsigned int *x) {
    *x = (*x>>24)|((*x<<8)&0x00FF0000)|((*x>>8)&0x0000FF00)|(*x<<24);
}

float * _Nullable * _Nullable readBinaryFile(const char * _Nonnull file, size_t * _Nonnull len1, size_t * _Nonnull len2, bool testData) {
    
    FILE *fimage = fopen(file, "rb");
    FILE *flabel = NULL;
    if (!fimage) {
        return NULL;
    } else {
        // We assume that the training set labels are located in the same directory as the training set images
        // So replace "train-images-idx3-ubyte" by "train-labels-idx1-ubyte" in the original path
        // Same assumption for the test set labels, replace t"10k-images-idx3-ubyte" by "t10k-labels-idx1-ubyte"
        // If we don't find the labels, we fatal...
        char *labelFile = malloc(strlen(file)*sizeof(char));
        memset(labelFile, 0, strlen(file)*sizeof(char));
        char *word = NULL;
        if (!testData) {
            word = "train";
        } else {
            word = "t10k";
        }
        char *ptr = strstr(file, word);
        if (ptr != NULL) {
            if (!testData) {
                fprintf(stdout, "%s: file path to MNIST training images file: %zu characters length.\n", PROGRAM_NAME, strlen(file));
            } else fprintf(stdout, "%s: file path to MNIST test images file: %zu characters length.\n", PROGRAM_NAME, strlen(file));
            size_t firstCopyLength = strlen(file) - strlen(ptr);
            memcpy(labelFile, file, firstCopyLength*sizeof(char));
            if (!testData) {
             strcat(labelFile, "train-labels-idx1-ubyte");
            } else strcat(labelFile, "t10k-labels-idx1-ubyte");
        } else {
            if (!testData) {
                fatal(PROGRAM_NAME, "MNIST training set images file not valid.");
            } else fatal(PROGRAM_NAME, "MNIST test set images file not valid.");
        }
        flabel = fopen(labelFile, "rb");
        if (!flabel) {
            if (!testData) {
                fatal(PROGRAM_NAME, "Can't find the training set labels file \'train-labels-idx1-ubyte\'.");
            } else fatal(PROGRAM_NAME, "Can't find the test set labels file \'t10k-labels-idx1-ubyte\'.");
        }
        if (!testData) {
            fprintf(stdout, "%s: got the training set labels.\n", PROGRAM_NAME);
        } else fprintf(stdout, "%s: got the test set labels.\n", PROGRAM_NAME);
        free(labelFile);
    }
    
    unsigned int magic, num, row, col;
    // Check if magic numbers are valid
    fread(&magic, 4, 1, fimage);
    if (magic != 0x03080000) fatal(PROGRAM_NAME, "magic number in traning/test set images file not correct.");
    
    fread(&magic, 4, 1, flabel);
    if (magic != 0x01080000) fatal(PROGRAM_NAME, "magic number in traning/test set labels file not correct.");
    
    fread(&num, 4, 1, flabel); // Just advance in this file
    fread(&num, 4, 1, fimage); endianSwap(&num);
    fread(&row, 4, 1, fimage); endianSwap(&row);
    fread(&col, 4, 1, fimage); endianSwap(&col);
    
    if (!testData) {
        fprintf(stdout,"%s: number of examples in MNIST training set: %d\n", PROGRAM_NAME, num);
        fprintf(stdout,"%s: number of features in each MNIST example: %d x %d\n", PROGRAM_NAME, col, row);
    } else fprintf(stdout,"%s: number of examples in MNIST test set: %d\n", PROGRAM_NAME, num);
    
    // Return a design matrix of the data set
    *len1 = num;
    *len2 = row*col + 1; // Number of features plus the ground-truth label
    float **dataSet = floatmatrix(0, *len1-1, 0, *len2-1);
    memset(*dataSet, 0.0f, (*len1*(*len2))*sizeof(float));
    int idx;
    for (int ex=0; ex<num; ex++) {
        idx = 0;
        for (int i=0; i<row; i++) {
            for (int j=0; j<col; j++) {
                unsigned char pixel;
                fread(&pixel, 1, 1, fimage);
                dataSet[ex][idx] = (float)pixel;
                idx++;
            }
        }
        unsigned char label;
        fread(&label, 1, 1, flabel);
        dataSet[ex][idx] = (float)label;
    }
    
    // Just show a few examples (here 10) from the dataset to check if we got the data properly
    // Output in hexadecimal
    fprintf(stdout, "---------------------------------------------------------------------\n");
    if (!testData) {
        fprintf(stdout, "Sample of the MNIST training data set.\n");
    } else fprintf(stdout, "Sample of the MNIST test data set.\n");
    fprintf(stdout, "---------------------------------------------------------------------\n");
    for (int ex=0; ex<10; ex++) {
        printf("---\n");
        idx = 0;
        for (int i=0; i<row*col; i++) {
            int byte = (int)dataSet[ex][i];
            printf("%02x", byte);
            idx++;
            if (idx == col) {
                idx = 0;
                printf("\n");
            }
        }
        printf("\n");
        printf("label = %d\n", (int)dataSet[ex][row*col]);
    }
    
    return dataSet;
}

float * _Nonnull * _Nonnull loadMnist(const char * _Nonnull file, size_t * _Nonnull len1, size_t * _Nonnull len2) {
    
    float **dataSet = readBinaryFile(file, len1, len2, false);
    if (dataSet == NULL) {
        fatal(PROGRAM_NAME, "problem reading MNIST data set.");
    } else fprintf(stdout, "%s: done.\n", PROGRAM_NAME);
    
    return dataSet;
}

float * _Nonnull * _Nonnull loadMnistTest(const char * _Nonnull file, size_t * _Nonnull len1, size_t * _Nonnull len2) {
    
    float **dataSet = readBinaryFile(file, len1, len2, true);
    if (dataSet == NULL) {
        fatal(PROGRAM_NAME, "problem reading the MNIST test data set ");
    } else fprintf(stdout, "%s: done.\n", PROGRAM_NAME);
    
    return dataSet;
}
