//
//  main.c
//  FeedforwardNT
//
//  Created by Seddik hakime on 21/05/2017.
//

#include "NeuralNetwork.h"

int main(int argc, const char * argv[]) {
    
    float **dataSet = NULL;
    float **trainingData = NULL, **testData = NULL, **validationData = NULL;
    
    char dataSetName[256];
    char dataSetFile[256];
    char testSetFile[256];
    int epochs, miniBatchSize, ntLayers[100], dataDivisions[2], classifications[100], inoutSizes[2];
    float eta, lambda;
    
    size_t len1=0, len2=0, tr1=0, tr2=0, ts1=0, ts2=0, tv1=0, tv2=0;
    size_t numberOfLayers=0, numberOfDataDivisions=0, numberOfClassifications=0, numberOfInouts=0;
    
    typedef struct testing {
        float * _Nonnull * _Nonnull (* _Nullable create) (float * _Nonnull * _Nonnull dataSet, size_t len1, size_t len2, size_t start, size_t end, size_t * _Nonnull t1, size_t * _Nonnull t2);
    } testing;
    typedef struct validation {
        float * _Nonnull * _Nonnull (* _Nullable create) (float * _Nonnull * _Nonnull dataSet, size_t len1, size_t len2, size_t start, size_t end, size_t * _Nonnull t1, size_t * _Nonnull t2);
    } validation;
    testing *test = NULL;
    validation *validate = NULL;
    
    if (argc < 2) {
        fatal(PROGRAM_NAME, "missing argument for the input parameters file.");
    }
    
    fprintf(stdout, "%s: start....\n", PROGRAM_NAME);
    
    bool pthread = false;
    bool availableTestData = false;
    bool err = false;
    
    if (argc >= 3 && argc < 4) {
        if (strcmp(argv[2], "-pthread") == 0) {
            pthread = true;
        } else if (strcmp(argv[2], "-test-data") == 0) {
            fatal(PROGRAM_NAME, "-test-data option given but no argument for location of test data set file present.");
        } else fatal(PROGRAM_NAME, "problem in argument list. Possibly unrecognized argument.");
    } else if (argc >= 3 && argc < 5) {
        if (strcmp(argv[2], "-pthread") == 0 || strcmp(argv[3], "-pthread") == 0) {
            fatal(PROGRAM_NAME, "-pthread and -test-data given but missing argument for location of test data set file.");
        } else {
            if (strcmp(argv[2], "-test-data") == 0 || strcmp(argv[3], "-test-data") == 0) {
                if (strcmp(argv[2], "-test-data") == 0) {
                    strncpy(testSetFile, argv[3], 256);
                } else if (strcmp(argv[3], "-test-data") == 0) {
                    strncpy(testSetFile, argv[2], 256);
                }
                availableTestData = true;
            } else {
                fatal(PROGRAM_NAME, "problem in argument list. Possibly unrecognized argument.");
            }
        }
    } else if (argc >= 3 && argc <= 5) {
        if (strcmp(argv[2], "-pthread") == 0) {
            pthread = true;
            if (strcmp(argv[3], "-test-data") == 0) {
                strncpy(testSetFile, argv[4], 256);
            } else if (strcmp(argv[4], "-test-data") == 0) {
                strncpy(testSetFile, argv[3], 256);
            } else {
                err = true;
            }
            if (!err) availableTestData = true;
        } else if (strcmp(argv[3], "-pthread") == 0) {
            pthread = true;
            if (strcmp(argv[2], "-test-data") == 0) {
                strncpy(testSetFile, argv[4], 256);
            } else if (strcmp(argv[4], "-test-data") == 0) {
                strncpy(testSetFile, argv[2], 256);
            } else {
                err = true;
            }
            if (!err) availableTestData = true;
        } else if (strcmp(argv[4], "-pthread") == 0) {
            pthread = true;
            if (strcmp(argv[2], "-test-data") == 0) {
                strncpy(testSetFile, argv[3], 256);
            } else if (strcmp(argv[3], "-test-data") == 0) {
                strncpy(testSetFile, argv[2], 256);
            } else {
                err = true;
            }
            if (!err) availableTestData = true;
        } else fatal(PROGRAM_NAME, "problem in argument list. Possibly unrecognized argument.");
    }
    
    if (err) fatal(PROGRAM_NAME, "problem in argument list. Possibly unrecognized argument.");

    if (pthread) {
        fprintf(stdout, "%s: multithreaded batch active.\n", PROGRAM_NAME);
    }
    if (availableTestData) {
        fprintf(stdout, "%s: used test data from test data set.\n", PROGRAM_NAME);
        validate = (validation *)malloc(sizeof(validation));
        validate->create = getData;
    } else {
        test = (testing *)malloc(sizeof(testing));
        test->create = getData;
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
    
    if (availableTestData) {
        fprintf(stdout, "\n");
        testData = loadTestData(dataSetName, testSetFile, &ts1, &ts2);
        // Create a validation data set from the training set
        validationData = validate->create(dataSet, len1, len2, dataDivisions[0], dataDivisions[1], &tv1, &tv2);
    } else {
        // No test data set available, create the test data from the training set
        testData = test->create(dataSet, len1, len2, dataDivisions[0], dataDivisions[1], &ts1, &ts2);
    }
    
    free_fmatrix(dataSet, 0, len1-1, 0, len2-1);
    if (test != NULL) free(test);
    if (validate != NULL)free(validate);
    
    NeuralNetwork *neural = allocateNeuralNetwork();
    neural->create((void *)neural, ntLayers, numberOfLayers, &miniBatchSize);
    
    fprintf(stdout, "%s: train neural network with the %s data set.\n", PROGRAM_NAME, dataSetName);
#ifdef COMPUTE_TOTAL_COST
    bool showCost = true;
#else
    bool showCost = false;
#endif
    neural->SDG((void *)neural, trainingData, testData, tr1, tr2, &ts1, &ts2, ntLayers, numberOfLayers, inoutSizes, classifications, epochs, miniBatchSize, eta, lambda, &showCost);
    neural->destroy((void *)neural);
    fprintf(stdout, "%s: all done.\n", PROGRAM_NAME);
    
    free(neural);
    free_fmatrix(trainingData, 0, tr1-1, 0, tr2-1);
    free_fmatrix(testData, 0, ts1-1, 0, ts2-1);
    if (validationData != NULL) free_fmatrix(validationData, 0, tv1-1, 0, tv2-1);
    
    return 0;
}
