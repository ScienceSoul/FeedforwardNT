//
//  main.c
//  FeedforwardNT
//
//  Created by Seddik hakime on 21/05/2017.
//

#include "NeuralNetwork.h"

int main(int argc, const char * argv[]) {
    
    float **dataSet = NULL;
    float **trainingData = NULL, **testData = NULL;
    
    char dataSetName[256];
    char dataSetFile[256];
    int epochs, miniBatchSize, ntLayers[100], dataDivisions[2], classifications[100], inoutSizes[2];
    float eta, lambda;
    
    size_t len1=0, len2=0, tr1=0, tr2=0, ts1=0, ts2=0;
    size_t numberOfLayers=0, numberOfDataDivisions=0, numberOfClassifications=0, numberOfInouts=0;
    
    if (argc < 2) {
        fatal(PROGRAM_NAME, "missing argument for the input parameters file.");
    }
    
    fprintf(stdout, "%s: start....\n", PROGRAM_NAME);
    
    bool pthread = false;
    if (argc == 3) {
        if (strcmp(argv[2], "-pthread") == 0) {
            pthread = true;
        }
    }
    if (pthread) {
        fprintf(stdout, "%s: multithreaded batch active.\n", PROGRAM_NAME);
    }
    
    memset(dataSetName, 0, sizeof(dataSetName));
    memset(dataSetFile, 0, sizeof(dataSetFile));
    
    memset(ntLayers, 0, sizeof(ntLayers));
    memset(dataDivisions, 0, sizeof(dataDivisions));
    memset(classifications, 0, sizeof(classifications));
    memset(inoutSizes, 0, sizeof(inoutSizes));
    
    fprintf(stdout, "%s: load input parameters:\n", PROGRAM_NAME);
    if (loadParameters(argv[1], dataSetName, dataSetFile, ntLayers, &numberOfLayers, dataDivisions, &numberOfDataDivisions, classifications, &numberOfClassifications, inoutSizes, &numberOfInouts, &epochs, &miniBatchSize, &eta, &lambda) != 0) {
        fatal(PROGRAM_NAME, "failure reading input parameters.");
    }
    fprintf(stdout, "%s: done.\n", PROGRAM_NAME);
    
    dataSet = loadData(dataSetName, dataSetFile, &len1, &len2);
    
    trainingData = createTrainigData(dataSet, 0, dataDivisions[0], &tr1, &tr2, classifications, numberOfClassifications, inoutSizes);
    testData = createTestData(dataSet, len1, len2, dataDivisions[0], dataDivisions[1], &ts1, &ts2);
    free_fmatrix(dataSet, 0, len1-1, 0, len2-1);
    
    NeuralNetwork *neural = allocateNeuralNetwork();
    neural->create((void *)neural, ntLayers, numberOfLayers, &miniBatchSize, pthread);
    
    fprintf(stdout, "%s: train neural network with the %s data set.\n", PROGRAM_NAME, dataSetName);
    bool showCost = true;
    neural->SDG((void *)neural, trainingData, testData, tr1, tr2, &ts1, &ts2, ntLayers, numberOfLayers, inoutSizes, classifications, epochs, miniBatchSize, eta, lambda, pthread, &showCost);
    neural->destroy((void *)neural, &miniBatchSize, pthread);
    fprintf(stdout, "%s: all done.\n", PROGRAM_NAME);
    
    free(neural);
    free_fmatrix(trainingData, 0, tr1-1, 0, tr2-1);
    free_fmatrix(testData, 0, ts1-1, 0, ts2-1);
    
    return 0;
}
