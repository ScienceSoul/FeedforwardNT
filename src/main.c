//
//  main.c
//  FeedforwardNT
//
//  Created by Seddik hakime on 21/05/2017.
//

#include "NeuralNetwork.h"
#include "LoadIrisDataSet.h"
#include "LoadMNISTDataSet.h"

bool metal = false;

int main(int argc, const char * argv[]) {
    
    char testFile[1024];
    
    if (argc < 2) {
        fatal(PROGRAM_NAME, "missing argument for the input parameters file.");
    }
    fprintf(stdout, "%s: start....\n", PROGRAM_NAME);
    
    bool availableTestData = false;
    bool err = false;
    
    memset(testFile, 0, sizeof(testFile));
    if (argc >= 3 && argc < 4) {
        if (strcmp(argv[2], "-metal") == 0) {
            metal = true;
        } else if (strcmp(argv[2], "-test-data") == 0) {
            fatal(PROGRAM_NAME, "-test-data option given but no argument for location of test data set file present.");
        } else fatal(PROGRAM_NAME, "problem in argument list. Possibly unrecognized argument.");
    } else if (argc >= 3 && argc < 5) {
        if (strcmp(argv[2], "-metal") == 0 || strcmp(argv[3], "-metal") == 0) {
            fatal(PROGRAM_NAME, "-metal and -test-data given but missing argument for location of test data set file.");
        } else {
            if (strcmp(argv[2], "-test-data") == 0 || strcmp(argv[3], "-test-data") == 0) {
                if (strcmp(argv[2], "-test-data") == 0) {
                    strncpy(testFile, argv[3], 256);
                } else if (strcmp(argv[3], "-test-data") == 0) {
                    strncpy(testFile, argv[2], 256);
                }
                availableTestData = true;
            } else {
                fatal(PROGRAM_NAME, "problem in argument list. Possibly unrecognized argument.");
            }
        }
    } else if (argc >= 3 && argc <= 5) {
        if (strcmp(argv[2], "-metal") == 0) {
            metal = true;
            if (strcmp(argv[3], "-test-data") == 0) {
                strncpy(testFile, argv[4], 256);
            } else if (strcmp(argv[4], "-test-data") == 0) {
                strncpy(testFile, argv[3], 256);
            } else {
                err = true;
            }
            if (!err) availableTestData = true;
        } else if (strcmp(argv[3], "-metal") == 0) {
            metal = true;
            if (strcmp(argv[2], "-test-data") == 0) {
                strncpy(testFile, argv[4], 256);
            } else if (strcmp(argv[4], "-test-data") == 0) {
                strncpy(testFile, argv[2], 256);
            } else {
                err = true;
            }
            if (!err) availableTestData = true;
        } else if (strcmp(argv[4], "-metal") == 0) {
            metal = true;
            if (strcmp(argv[2], "-test-data") == 0) {
                strncpy(testFile, argv[3], 256);
            } else if (strcmp(argv[3], "-test-data") == 0) {
                strncpy(testFile, argv[2], 256);
            } else {
                err = true;
            }
            if (!err) availableTestData = true;
        } else fatal(PROGRAM_NAME, "problem in argument list. Possibly unrecognized argument.");
    }
    
    if (err) fatal(PROGRAM_NAME, "problem in argument list. Possibly unrecognized argument.");
    
    //memset(dataSetName, 0, sizeof(dataSetName));
    //memset(trainFile, 0, sizeof(trainFile));
    
    // Instantiate a neural network and load its parameters...
    fprintf(stdout, "%s: load the network and its input parameters:\n", PROGRAM_NAME);
    NeuralNetwork *neural = newNeuralNetwork();
    if (neural->load((void *)neural, argv[1]) != 0) {
        fatal(PROGRAM_NAME, "failure reading input parameters.");
    }
    fprintf(stdout, "%s: done.\n", PROGRAM_NAME);
    
    // Create the data structures of the neural network
    fprintf(stdout, "%s: create the network internal structure:\n", PROGRAM_NAME);
    neural->genesis((void *)neural);
    
    // Allocate and initialize the network data containers
    neural->data->init((void *)neural);
    
    if (strcmp(neural->parameters->dataName, "iris") == 0) {
        neural->data->training->reader = loadIris;
    } else if (strcmp(neural->parameters->dataName, "mnist") == 0) {
        neural->data->training->reader = loadMnist;
    } else {
        fatal(PROGRAM_NAME, "Program can only train for Iris or MNIST data sets.");
    }
    
    if (availableTestData) {
        if (strcmp(neural->parameters->dataName, "mnist") != 0) fatal(PROGRAM_NAME, "Program can only use MNIST test data.");
        fprintf(stdout, "%s: use test data from test data set.\n", PROGRAM_NAME);
        neural->data->test->reader = loadMnistTest;
    }
    
    // Load all training/test data
    neural->data->load((void *)neural, neural->parameters->dataName, neural->parameters->data, testFile, availableTestData);
    fprintf(stdout, "%s: done.\n", PROGRAM_NAME);
    
    fprintf(stdout, "%s: train neural network with the %s data set.\n", PROGRAM_NAME, neural->parameters->dataName);
#ifdef COMPUTE_TOTAL_COST
    bool showCost = true;
#else
    bool showCost = false;
#endif
    
#ifdef __APPLE__
    if (metal) {
        neural->gpu_alloc((void *)neural);
        neural->gpu->init();
    }
#endif
    neural->compute((void *)neural, &showCost);
    neural->finale((void *)neural);
    fprintf(stdout, "%s: all done.\n", PROGRAM_NAME);
    free(neural);
    
    return 0;
}
