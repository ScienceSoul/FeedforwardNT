//
//  NeuralNetwork.c
//  FeedforwardNT
//
//  Created by Seddik hakime on 31/05/2017.
//

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include "cblas.h"
    #include "cblas_f77.h"
#endif

#include "NeuralNetwork.h"

static int loadParameters(void * _Nonnull self, const char * _Nonnull paraFile, char * _Nonnull dataSetName, char * _Nonnull dataSetFile);

static void initNeuralData(void * _Nonnull self);
static void loadData(void * _Nonnull self, const char * _Nonnull dataSetName, const char * _Nonnull trainFile, const char * _Nonnull testFile, bool testData);

static float * _Nonnull * _Nonnull createTrainigData(float * _Nonnull * _Nonnull dataSet, unsigned int start, unsigned int end, unsigned int * _Nonnull t1, unsigned int * _Nonnull t2, int * _Nonnull classifications, unsigned int numberOfClassifications, int * _Nonnull ntLayers, int numberOfLayers);

static float * _Nonnull * _Nonnull getData(float * _Nonnull * _Nonnull dataSet, unsigned int len1, unsigned int len2, unsigned int start, unsigned int end, unsigned int * _Nonnull t1, unsigned int * _Nonnull t2);

static activationNode * _Nonnull allocateActivationNode(void);
static zNode * _Nonnull allocateZNode(void);

static dcdwNode * _Nonnull allocateDcdwNode(void);
static dcdbNode * _Nonnull allocateDcdbNode(void);

static float * _Nonnull initWeights(int * _Nonnull ntLayers, unsigned int numberOfLayers);
static float * _Nonnull initBiases(int * _Nonnull ntLayers, unsigned int numberOfLayers);

static activationNode * _Nonnull initActivationsList(int * _Nonnull ntLayers, unsigned int numberOfLayers);
static zNode * _Nonnull initZsList(int * _Nonnull ntLayers, unsigned int numberOfLayers);

static dcdwNode * _Nonnull initDcdwList(int * _Nonnull ntLayers, unsigned int numberOfLayers);
static dcdbNode * _Nonnull initDcdbList(int * _Nonnull ntLayers, unsigned int numberOfLayers);

static void genesis(void * _Nonnull self);
static void finale(void * _Nonnull self);
static void gpu_alloc(void * _Nonnull self);

static void computeNeural(void * _Nonnull self, bool * _Nullable showTotalCost);

static void miniBatch(void * _Nonnull self, float * _Nonnull * _Nonnull miniBatch);

static void updateWeightsBiases(void * _Nonnull self);

static void batchAccumulation(void * _Nonnull self);

static void * _Nullable backpropagation(void * _Nonnull self);

static int evaluate(void * _Nonnull self);

static float totalCost(void * _Nonnull self, float * _Nonnull * _Nonnull data, unsigned int m, bool convert);

static void feedforward(void * _Nonnull self);

static int loadParameters(void * _Nonnull self, const char * _Nonnull paraFile, char * _Nonnull dataSetName, char * _Nonnull dataSetFile) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
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
            parseArgument(string, "network definition", nn->parameters->ntLayers, &nn->parameters->numberOfLayers);
        }
        if (lineCount == 5) {
            parseArgument(string, "data divisions", nn->parameters->dataDivisions, &nn->parameters->numberOfDataDivisions);
        }
        if (lineCount == 6) {
            parseArgument(string, "classifications", nn->parameters->classifications, &nn->parameters->numberOfClassifications);
        }
        if (lineCount == 7) {
            nn->parameters->epochs = atoi(string);
        }
        if (lineCount == 8) {
            nn->parameters->miniBatchSize = atoi(string);
        }
        if (lineCount == 9) {
            nn->parameters->eta = strtof(string, NULL);
        }
        if (lineCount == 10) {
            nn->parameters->lambda = strtof(string, NULL);
        }
        lineCount++;
    }
    
    if (nn->parameters->numberOfDataDivisions != 2) {
        fatal(PROGRAM_NAME, " input data set should only be divided in two parts: one for training, one for testing.");
    }
    
    return 0;
}

static void initNeuralData(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    nn->data->training = (training *)malloc(sizeof(training));
    nn->data->training->set = NULL;
    nn->data->training->m = 0;
    nn->data->training->n = 0;
    
    nn->data->test = (test *)malloc(sizeof(test));
    nn->data->test->set = NULL;
    nn->data->test->m = 0;
    nn->data->test->n = 0;
    
    nn->data->validation = (validation *)malloc(sizeof(validation));
    nn->data->validation->set = NULL;
    nn->data->validation->m = 0;
    nn->data->validation->n = 0;
}

static void loadData(void * _Nonnull self, const char * _Nonnull dataSetName, const char * _Nonnull trainFile, const char * _Nonnull testFile, bool testData) {
    
    unsigned int len1=0, len2=0;
    float **raw_training = NULL;
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    fprintf(stdout, "%s: load the <%s> training data set.\n", PROGRAM_NAME, dataSetName);
    raw_training = nn->data->training->reader(trainFile, &len1, &len2);
    shuffle(raw_training, len1, len2);
    
    nn->data->training->set = createTrainigData(raw_training, 0, nn->parameters->dataDivisions[0], &nn->data->training->m, &nn->data->training->n, nn->parameters->classifications, nn->parameters->numberOfClassifications, nn->parameters->ntLayers, nn->parameters->numberOfLayers);
    
    if (testData) {
        nn->data->test->set = nn->data->test->reader(testFile, &nn->data->test->m, &nn->data->test->n);
        nn->data->validation->set = getData(raw_training, len1, len2, nn->parameters->dataDivisions[0], nn->parameters->dataDivisions[1], &nn->data->validation->m, &nn->data->validation->n);
    } else {
        nn->data->test->set = getData(raw_training, len1, len2, nn->parameters->dataDivisions[0], nn->parameters->dataDivisions[1], &nn->data->test->m, &nn->data->test->n);
    }
}

static float * _Nonnull * _Nonnull createTrainigData(float * _Nonnull * _Nonnull dataSet, unsigned int start, unsigned int end, unsigned int * _Nonnull t1, unsigned int * _Nonnull t2, int * _Nonnull classifications, unsigned int numberOfClassifications, int * _Nonnull ntLayers, int numberOfLayers) {
    
    float **trainingData = NULL;
    trainingData = floatmatrix(0, end-1, 0, (ntLayers[0]+ntLayers[numberOfLayers-1])-1);
    *t1 = end;
    *t2 = ntLayers[0]+ntLayers[numberOfLayers-1];
    
    if (ntLayers[numberOfLayers-1] != numberOfClassifications) {
        fatal(PROGRAM_NAME, "the number of classifications should be equal to the number of activations at the output layer.");
    }
    
    for (int i=0; i<end; i++) {
        for (int j=0; j<ntLayers[0]; j++) {
            trainingData[i][j] = dataSet[i][j];
        }
        
        // Binarization of t    he input ground-truth to get a one-hot-vector
        for (int k=0; k<numberOfClassifications; k++) {
            if (dataSet[i][ntLayers[0]] == (float)classifications[k]) {
                trainingData[i][ntLayers[0]+k] = 1.0f;
            } else trainingData[i][ntLayers[0]+k] = 0.0f;
        }
    }
    
    return trainingData;
}

static float * _Nonnull * _Nonnull getData(float * _Nonnull * _Nonnull dataSet, unsigned int len1, unsigned int len2, unsigned int start, unsigned int end, unsigned int * _Nonnull t1, unsigned int * _Nonnull t2) {
    
    float **data = floatmatrix(0, end, 0, len2-1);
    *t1 = end;
    *t2 = len2;
    
    int idx = 0;
    for (int i=(int)start; i<start+end; i++) {
        for (int j=0; j<len2; j++) {
            data[idx][j] = dataSet[i][j];
        }
        idx++;
    }
    return data;
}

static activationNode * _Nonnull allocateActivationNode(void) {
    activationNode *list = (activationNode *)malloc(sizeof(activationNode));
    *list = (activationNode){.n=0, .a=NULL, .next=NULL, .previous=NULL};
    return list;
}

static zNode * _Nonnull allocateZNode(void) {
    zNode *list = (zNode *)malloc(sizeof(zNode));
    *list = (zNode){.n=0, .z=NULL, .next=NULL, .previous=NULL};
    return list;
}

static dcdwNode * _Nonnull allocateDcdwNode(void) {
    dcdwNode *list = (dcdwNode *)malloc(sizeof(dcdwNode));
    *list = (dcdwNode){.m=0, .n=0, .dcdw=NULL, .next=NULL, .previous=NULL};
    return list;
}

static dcdbNode * _Nonnull allocateDcdbNode(void) {
    dcdbNode *list = (dcdbNode *)malloc(sizeof(dcdbNode));
    *list = (dcdbNode){.n=0, .dcdb=NULL, .next=NULL, .previous=NULL};
    return list;
}

//
//  Create the weights vector according to the number of layers in the network.
//  The weights are initialized using a Gaussian distribution with mean 0
//  and standard deviation 1 over the square root of the number of
//  weights connecting to the same neuron.
//
static float * _Nonnull initWeights(int * _Nonnull ntLayers, unsigned int numberOfLayers) {
    
    int dim = 0;
    for (int l=0; l<numberOfLayers-1; l++) {
        dim = dim + (ntLayers[l+1]*ntLayers[l]);
    }
    float *weights = (float *)malloc(dim*sizeof(float));
    
    int stride = 0;
    for (int l=0; l<numberOfLayers-1; l++) {
         int m = ntLayers[l+1];
         int n = ntLayers[l];
         for (int i = 0; i<m; i++) {
             for (int j=0; j<n; j++) {
                 weights[stride+((i*n)+j)] = randn(0.0f, 1.0f) / sqrtf((float)n);
             }
         }
         stride = stride + (m * n);
     }
    
    return weights;
}

//
//  Create the biases vector according to the number of layers in the network.
//  The biases are initialized using a Gaussian distribution with mean 0
//  and standard deviation 1.
//
static float * _Nonnull initBiases(int * _Nonnull ntLayers, unsigned int numberOfLayers) {
    
    int dim = 0;
    for (int l=1; l<numberOfLayers; l++) {
        dim  = dim + ntLayers[l];
    }
    
    float *biases = (float*)malloc(dim*sizeof(float));
    
    int stride = 0;
    for (int l=1; l<numberOfLayers; l++) {
        int n = ntLayers[l];
        for (int i = 0; i<n; i++) {
            biases[stride+i] = randn(0.0f, 1.0f);
        }
        stride = stride + n;
    }
    
    return biases;
}

//
//  Create the activations list according to the number of layers in the network.
//
//  Return a pointer to the list head.
//
static activationNode * _Nonnull initActivationsList(int * _Nonnull ntLayers, unsigned int numberOfLayers) {
    
    activationNode *activationsList = allocateActivationNode();
    
    // The first activation node (i.e., layer)
    activationsList->a = floatvec(0, ntLayers[0]-1);
    activationsList->n = ntLayers[0];
    // The rest of the activation nodes (i.e., layers)
    int idx = 1;
    int k = 1;
    activationNode *aNodePt = activationsList;
    while (k <= numberOfLayers-1) {
        activationNode *newNode = allocateActivationNode();
        newNode->a = floatvec(0, ntLayers[idx]-1);
        newNode->n = ntLayers[idx];
        newNode->previous = aNodePt;
        aNodePt->next = newNode;
        aNodePt = newNode;
        k++;
        idx++;
    }
    
    aNodePt = activationsList;
    while (aNodePt != NULL) {
        memset(aNodePt->a, 0.0f, aNodePt->n*sizeof(float));
        aNodePt = aNodePt->next;
    }
    
    return activationsList;
}

//
//  Create the zs list according to the number of layers in the network.
//
//  Return a pointer to the list head.
//
static zNode * _Nonnull initZsList(int * _Nonnull ntLayers, unsigned int numberOfLayers) {
    
    zNode *zsList = allocateZNode();
    
    // The first z node (i.e., layer)
    zsList->z = floatvec(0, ntLayers[0]-1);
    zsList->n = ntLayers[0];
    // The rest of the z nodes (i.e., layers)
    int idx = 1;
    int k = 1;
    zNode *zNodePt = zsList;
    while (k <= numberOfLayers-1) {
        zNode *newNode = allocateZNode();
        newNode->z = floatvec(0, ntLayers[idx]-1);
        newNode->n = ntLayers[idx];
        newNode->previous = zNodePt;
        zNodePt->next = newNode;
        zNodePt = newNode;
        k++;
        idx++;
    }
    
    zNodePt = zsList;
    while (zNodePt != NULL) {
        memset(zNodePt->z, 0.0f, zNodePt->n*sizeof(float));
        zNodePt = zNodePt->next;
    }
    
    return zsList;
}

//
//  Create the dC/dw list according to the number of layers in the network.
//
//  Return a pointer to the list head.
//
static dcdwNode * _Nonnull initDcdwList(int * _Nonnull ntLayers, unsigned int numberOfLayers) {
    
    dcdwNode *dcdwList = allocateDcdwNode();
    
    // The first weight node (i.e., layer)
    dcdwList->dcdw = floatmatrix(0, ntLayers[1]-1, 0, ntLayers[0]-1);
    dcdwList->m = ntLayers[1];
    dcdwList->n = ntLayers[0];
    // The rest of the weight nodes (i.e., layers)
    int idx = 1;
    int k = 1;
    dcdwNode *dcdwNodePt = dcdwList;
    while (k < numberOfLayers-1) {
        dcdwNode *newNode = allocateDcdwNode();
        newNode->dcdw = floatmatrix(0, ntLayers[idx+1]-1, 0, ntLayers[idx]-1);
        newNode->m = ntLayers[idx+1];
        newNode->n = ntLayers[idx];
        newNode->previous = dcdwNodePt;
        dcdwNodePt->next = newNode;
        dcdwNodePt = newNode;
        k++;
        idx++;
    }
    
    return dcdwList;
}

//
//  Create the dC/db list according to the number of layers in the network.
//
//  Return a pointer to the list head.
//
static dcdbNode * _Nonnull initDcdbList(int * _Nonnull ntLayers, unsigned int numberOfLayers) {
    
    dcdbNode *dcdbList = allocateDcdbNode();
    
    // The first bias node (i.e., layer)
    dcdbList->dcdb = floatvec(0, ntLayers[1]-1);
    dcdbList->n = ntLayers[1];
    // The rest of the bias nodes (i.e., layers)
    int idx = 2;
    int k = 1;
    dcdbNode *dcdbNodePt = dcdbList;
    while (k < numberOfLayers-1) {
        dcdbNode *newNode = allocateDcdbNode();
        newNode->dcdb = floatvec(0, ntLayers[idx]-1);
        newNode->n = ntLayers[idx];
        newNode->previous = dcdbNodePt;
        dcdbNodePt->next = newNode;
        dcdbNodePt = newNode;
        k++;
        idx++;
    }
    
    return dcdbList;
}

//
// Allocate memory for a neural network
//
NeuralNetwork * _Nonnull newNeuralNetwork(void) {
    
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    *nn = (NeuralNetwork){.weights=NULL, .biases=NULL, .activationsList=NULL, .zsList=NULL,
                          .dcdwsList=NULL, .dcdbsList=NULL, .delta_dcdwsList=NULL, .delta_dcdbsList=NULL, .gpu=NULL};
    
    nn->parameters = (parameters *)malloc(sizeof(parameters));
    nn->parameters->epochs = 0;
    nn->parameters->miniBatchSize = 0;
    nn->parameters->eta = 0.0f;
    nn->parameters->lambda = 0.0f;
    memset(nn->parameters->ntLayers, 0, sizeof(nn->parameters->ntLayers));
    memset(nn->parameters->classifications, 0, sizeof(nn->parameters->classifications));
    memset(nn->parameters->dataDivisions, 0, sizeof(nn->parameters->dataDivisions));
    nn->load = loadParameters;
    
    nn->genesis = genesis;
    nn->finale = finale;
    nn->gpu_alloc = gpu_alloc;
    nn->compute = computeNeural;
    nn->miniBatch = miniBatch;
    nn->updateWeightsBiases = updateWeightsBiases;
    nn->batchAccumulation = batchAccumulation;
    nn->backpropagation = backpropagation;
    nn->evaluate = evaluate;
    nn->totalCost = totalCost;
    nn->feedforward = feedforward;
    
    return nn;
}

//
//  Create the network layers, i.e. allocates memory for the weight, bias, activation, z, dC/dx and dC/db data structures
//
static void genesis(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    nn->example_idx = 0;
    nn->number_of_features = 0;
    nn->number_of_layers = nn->parameters->numberOfLayers;
    nn->max_number_of_nodes_in_layer = max_array(nn->parameters->ntLayers, nn->parameters->numberOfLayers);
    
    nn->data = (data *)malloc(sizeof(data));
    nn->data->init = initNeuralData;
    nn->data->load = loadData;
    
    for (int l=0; l<nn->parameters->numberOfLayers-1; l++) {
        nn->weightsDimensions[l].m = nn->parameters->ntLayers[l+1];
        nn->weightsDimensions[l].n = nn->parameters->ntLayers[l];
    }
    for (int l=1; l<nn->parameters->numberOfLayers; l++) {
        nn->biasesDimensions[l-1].n = nn->parameters->ntLayers[l];
    }
    
    nn->weights = initWeights(nn->parameters->ntLayers, nn->parameters->numberOfLayers);
    nn->biases = initBiases(nn->parameters->ntLayers, nn->parameters->numberOfLayers);
    nn->activationsList = initActivationsList(nn->parameters->ntLayers, nn->parameters->numberOfLayers);
    nn->zsList = initZsList(nn->parameters->ntLayers, nn->parameters->numberOfLayers);
    nn->dcdwsList = initDcdwList(nn->parameters->ntLayers, nn->parameters->numberOfLayers);
    nn->dcdbsList = initDcdbList(nn->parameters->ntLayers, nn->parameters->numberOfLayers);
    nn->delta_dcdwsList = initDcdwList(nn->parameters->ntLayers, nn->parameters->numberOfLayers);
    nn->delta_dcdbsList = initDcdbList(nn->parameters->ntLayers, nn->parameters->numberOfLayers);
}

//
// Free-up all the memory used by a network
//
static void finale(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    free_fmatrix(nn->data->training->set, 0, nn->data->training->m, 0, nn->data->training->n);
    free_fmatrix(nn->data->test->set, 0, nn->data->test->m, 0, nn->data->test->n);
    if (nn->data->validation->set != NULL) free_fmatrix(nn->data->validation->set, 0, nn->data->validation->m, 0, nn->data->validation->n);
    nn->data->training->reader = NULL;
    nn->data->test->reader = NULL;
    free(nn->data->training);
    free(nn->data->test);
    free(nn->data->validation);
    nn->data->init = NULL;
    nn->data->load = NULL;
    free(nn->data);
    free(nn->parameters);
    
    free(nn->weights);
    free(nn->biases);
    
    dcdwNode *dcdwTail = nn->dcdwsList;
    while (dcdwTail != NULL && dcdwTail->next ) {
        dcdwTail = dcdwTail->next;
    }
    dcdwNode *dcdwNodePt = NULL;
    while (dcdwTail != NULL) {
        dcdwNodePt = dcdwTail->previous;
        free_fmatrix(dcdwTail->dcdw, 0, dcdwTail->m-1, 0, dcdwTail->n-1);
        dcdwTail->dcdw = NULL;
        dcdwTail->next = NULL;
        dcdwTail->previous = NULL;
        free(dcdwTail);
        dcdwTail = dcdwNodePt;
    }
    
    dcdwNode *delta_dcdwTail = nn->delta_dcdwsList;
    while (delta_dcdwTail != NULL && delta_dcdwTail->next ) {
        delta_dcdwTail = delta_dcdwTail->next;
    }
    dcdwNode *delta_dcdwNodePt = NULL;
    while (delta_dcdwTail != NULL) {
        delta_dcdwNodePt = delta_dcdwTail->previous;
        free_fmatrix(delta_dcdwTail->dcdw, 0, delta_dcdwTail->m-1, 0, delta_dcdwTail->n-1);
        delta_dcdwTail->dcdw = NULL;
        delta_dcdwTail->next = NULL;
        delta_dcdwTail->previous = NULL;
        free(delta_dcdwTail);
        delta_dcdwTail = delta_dcdwNodePt;
    }
    
    dcdbNode *dcdbTail = nn->dcdbsList;
    while (dcdbTail != NULL && dcdbTail->next != NULL) {
        dcdbTail = dcdbTail->next;
    }
    dcdbNode *dcdbNodePt = NULL;
    while (dcdbTail != NULL) {
        dcdbNodePt = dcdbTail->previous;
        free_fvector(dcdbTail->dcdb, 0, dcdbTail->n);
        dcdbTail->dcdb = NULL;
        dcdbTail->next = NULL;
        dcdbTail->previous = NULL;
        free(dcdbTail);
        dcdbTail = dcdbNodePt;
    }
    
    dcdbNode *delta_dcdbTail = nn->delta_dcdbsList;
    while (delta_dcdbTail != NULL && delta_dcdbTail->next != NULL) {
        delta_dcdbTail = delta_dcdbTail->next;
    }
    dcdbNode *delta_dcdbNodePt = NULL;
    while (delta_dcdbTail != NULL) {
        delta_dcdbNodePt = delta_dcdbTail->previous;
        free_fvector(delta_dcdbTail->dcdb, 0, delta_dcdbTail->n);
        delta_dcdbTail->dcdb = NULL;
        delta_dcdbTail->next = NULL;
        delta_dcdbTail->previous = NULL;
        free(delta_dcdbTail);
        delta_dcdbTail = delta_dcdbNodePt;
    }
    
    activationNode *aTail = nn->activationsList;
    while (aTail != NULL && aTail->next != NULL) {
        aTail = aTail->next;
    }
    activationNode *aNodePt = NULL;
    while (aTail != NULL) {
        aNodePt = aTail->previous;
        free_fvector(aTail->a, 0, aTail->n);
        aTail->a = NULL;
        aTail->next = NULL;
        aTail->previous = NULL;
        free(aTail);
        aTail = aNodePt;
    }
    
    zNode *zTail = nn->zsList;
    while (zTail != NULL && zTail->next != NULL) {
        zTail = zTail->next;
    }
    zNode *zNodePt = NULL;
    while (zTail != NULL) {
        zNodePt = zTail->previous;
        free_fvector(zTail->z, 0, zTail->n);
        zTail->z = NULL;
        zTail->next = NULL;
        zTail->previous = NULL;
        free(zTail);
        zTail = zNodePt;
    }
    
    if (nn->gpu != NULL) {
        nn->gpu->nullify();
        free(nn->gpu);
    };
}

static void gpu_alloc(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    nn->gpu = metalCompute();
}

static void computeNeural(void * _Nonnull self, bool * _Nullable showTotalCost) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    nn->number_of_features = nn->parameters->ntLayers[0];
    
    // Stochastic gradient descent
    float **miniBatch = floatmatrix(0, nn->parameters->miniBatchSize-1, 0, nn->data->training->n-1);
    nn->batch = miniBatch;
    int delta;
    for (int k=1; k<=nn->parameters->epochs; k++) {
        delta = 0;
        shuffle(nn->data->training->set, nn->data->training->m, nn->data->training->n);
        
        fprintf(stdout, "%s: Epoch {%d/%d}:\n", PROGRAM_NAME, k, nn->parameters->epochs);
        double train_time = 0.0;
        const int percentPrint = 5;
        int train_size = (int)nn->data->training->m/nn->parameters->miniBatchSize;
        int step = train_size / (100/percentPrint);
        int nextPrint = step;
        int i = 0;
        for (int l=1; l<=(int)nn->data->training->m/nn->parameters->miniBatchSize; l++) {
            memcpy(*miniBatch, *nn->data->training->set+delta, (nn->parameters->miniBatchSize*(int)nn->data->training->n)*sizeof(float));
            double rt = realtime();
            nn->miniBatch((void *)nn, miniBatch);
            rt = realtime() - rt;
            train_time += rt;
            delta = delta + (nn->parameters->miniBatchSize*(int)nn->data->training->n);
            
            i++;
            if (i >= nextPrint) {
                int percent = (100 * i) / train_size;
                fprintf(stdout, "...%d%%\n", percent);
                fflush(stdout);
                nextPrint += step;
            }
        }
        fprintf(stdout, "%s: time to complete all training data set (s): %f\n", PROGRAM_NAME, train_time);
        
        if (nn->data->test->set != NULL) {
            fprintf(stdout, "%s: Epoch {%d/%d}: testing network with {%u} inputs:\n", PROGRAM_NAME, k, nn->parameters->epochs, nn->data->test->m);
            int result = nn->evaluate(self);
            fprintf(stdout, "%s: Epoch {%d/%d}: {%d} / {%u}.\n", PROGRAM_NAME, k, nn->parameters->epochs, result, nn->data->test->m);
        }
        
        if (showTotalCost != NULL) {
            if (*showTotalCost == true) {
                double rt = realtime();
                float cost = nn->totalCost(self, nn->data->training->set, nn->data->training->m, false);
                rt = realtime() -  rt;
                fprintf(stdout, "%s: cost on training data: {%f} / Time (s): %f\n", PROGRAM_NAME, cost, rt);
                
                if (nn->data->test->set != NULL) {
                    double rt = realtime();
                    cost = nn->totalCost(self, nn->data->test->set, nn->data->test->m, true);
                    rt = realtime() -  rt;
                    fprintf(stdout, "%s: cost on test data: {%f} / Time (s): %f\n", PROGRAM_NAME, cost, rt);
                }
            }
        }
        fprintf(stdout, "\n");
    }

    free_fmatrix(miniBatch, 0, nn->parameters->miniBatchSize-1, 0, nn->data->training->n-1);
}

static void miniBatch(void * _Nonnull self, float * _Nonnull * _Nonnull miniBatch) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    dcdwNode *dcdwNodePt = nn->dcdwsList;
    while (dcdwNodePt != NULL) {
        memset(*dcdwNodePt->dcdw, 0.0f, (dcdwNodePt->m*dcdwNodePt->n)*sizeof(float));
        dcdwNodePt = dcdwNodePt->next;
    }
    
    dcdbNode *dcdbNodePt = nn->dcdbsList;
    while (dcdbNodePt != NULL) {
        memset(dcdbNodePt->dcdb, 0.0f, dcdbNodePt->n*sizeof(float));
        dcdbNodePt = dcdbNodePt->next;
    }
    
    dcdwNode *delta_dcdwNodePt = nn->delta_dcdwsList;
    while (delta_dcdwNodePt != NULL) {
        memset(*delta_dcdwNodePt->dcdw, 0.0f, (delta_dcdwNodePt->m*delta_dcdwNodePt->n)*sizeof(float));
        delta_dcdwNodePt = delta_dcdwNodePt->next;
    }
    
    dcdbNode *delta_dcdbNodePt = nn->delta_dcdbsList;
    while (delta_dcdbNodePt != NULL) {
        memset(delta_dcdbNodePt->dcdb, 0.0f, delta_dcdbNodePt->n*sizeof(float));
        delta_dcdbNodePt = delta_dcdbNodePt->next;
    }
    
    double rt = realtime();
    for (int i=0; i<nn->parameters->miniBatchSize; i++) {
        nn->example_idx = i;
        nn->backpropagation((void *)nn);
        nn->batchAccumulation((void *)nn);
    }
    rt = realtime() - rt;
#ifdef VERBOSE
    fprintf(stdout, "%s: time to complete a single mini-batch (s): %f\n", PROGRAM_NAME, rt);
#endif
    
    nn->updateWeightsBiases((void *)nn);
}

static void batchAccumulation(void * _Nonnull self) {
    
    // Accumulate dcdw and dc/db
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    dcdwNode *dcdwNodePt = nn->dcdwsList;
    dcdbNode *dcdbNodePt = nn->dcdbsList;
    dcdwNode *delta_dcdwNodePt = nn->delta_dcdwsList;
    dcdbNode *delta_dcdbNodePt = nn->delta_dcdbsList;
    while (dcdwNodePt != NULL && delta_dcdwNodePt != NULL) {
        for (int i=0; i<dcdwNodePt->m; i++) {
            for (int j=0; j<dcdwNodePt->n; j++) {
                dcdwNodePt->dcdw[i][j] = dcdwNodePt->dcdw[i][j] + delta_dcdwNodePt->dcdw[i][j];
            }
        }
        for (int i=0; i<dcdbNodePt->n; i++) {
            dcdbNodePt->dcdb[i] = dcdbNodePt->dcdb[i] + delta_dcdbNodePt->dcdb[i];
        }
        dcdwNodePt = dcdwNodePt->next;
        dcdbNodePt = dcdbNodePt->next;
        delta_dcdwNodePt = delta_dcdwNodePt->next;
        delta_dcdbNodePt = delta_dcdbNodePt->next;
    }
}

static void updateWeightsBiases(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    // Update weights
    unsigned int stride = 0;
    unsigned int l = 0;
    dcdwNode *dcdwNodePt = nn->dcdwsList;
    while (dcdwNodePt != NULL) {
        unsigned int m = nn->weightsDimensions[l].m;
        unsigned int n = nn->weightsDimensions[l].n;
        for (int i=0; i<m; i++) {
            for (int j=0; j<n; j++) {
                nn->weights[stride+((i*n)+j)] = (1.0f-((nn->parameters->eta*nn->parameters->lambda)/(float)nn->data->training->m))*nn->weights[stride+((i*n)+j)] - (nn->parameters->eta/(float)nn->parameters->miniBatchSize)*dcdwNodePt->dcdw[i][j];
            }
        }
        dcdwNodePt = dcdwNodePt->next;
        l++;
        stride = stride + (m * n);
    }
    
    // Update biases
    stride = 0;
    l = 0;
    dcdbNode *dcdbNodePt = nn->dcdbsList;
    while (dcdbNodePt != NULL) {
        unsigned int n = nn->biasesDimensions[l].n;
        for (int i=0; i<n; i++) {
            nn->biases[stride+i] = nn->biases[stride+i] - (nn->parameters->eta/(float)nn->parameters->miniBatchSize)*dcdbNodePt->dcdb[i];
        }
        dcdbNodePt = dcdbNodePt->next;
        l++;
        stride = stride + n;
    }
}

//
//  Return the gradient of the cross-entropy cost function C_x layers by layers
//
static void * _Nullable backpropagation(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    // Activations at the input layer
    activationNode *aNodePt = nn->activationsList;
    for (int i=0; i<nn->number_of_features; i++) {
        aNodePt->a[i] = nn->batch[nn->example_idx][i];
    }
    
    // Feedforward
    feedforward(nn);
    
    // ------------- Backward pass
    // At last layer
    
    activationNode *aTail = nn->activationsList;
    while (aTail != NULL && aTail->next != NULL) {
        aTail = aTail->next;
    }
    zNode *zTail = nn->zsList;
    while (zTail != NULL && zTail->next != NULL) {
        zTail = zTail->next;
    }
    
    float delta[nn->max_number_of_nodes_in_layer];
    float buffer[nn->max_number_of_nodes_in_layer];
    memset(delta, 0.0f, sizeof(delta));
    memset(buffer, 0.0f, sizeof(buffer));
    
    // Compute delta
    int k = (int)nn->number_of_features;
    for (int i=0; i<aTail->n; i++) {
        delta[i] = aTail->a[i] - nn->batch[nn->example_idx][k];
        k++;
    }
    
    //dc/dw and dc/db at last layer
    dcdwNode *dcdwTail = nn->delta_dcdwsList;
    while (dcdwTail != NULL && dcdwTail->next != NULL) {
        dcdwTail = dcdwTail->next;
    }
    dcdbNode *dcdbTail = nn->delta_dcdbsList;
    while (dcdbTail != NULL && dcdbTail->next != NULL) {
        dcdbTail = dcdbTail->next;
    }
    aNodePt = aTail->previous;
    for (int i=0; i<dcdwTail->m; i++) {
        for (int j=0; j<dcdwTail->n; j++) {
            dcdwTail->dcdw[i][j] = aNodePt->a[j]*delta[i];
        }
    }
    for (int i=0; i<dcdbTail->n; i++) {
        dcdbTail->dcdb[i] = delta[i];
    }
    
    // The backward pass loop
    
    // Stride to weithts at last layer
    unsigned int stride = 0;
    unsigned int m, n;
    for (int l=0; l<nn->parameters->numberOfLayers-2; l++) {
        m = nn->weightsDimensions[l].m;
        n = nn->weightsDimensions[l].n;
        stride = stride + (m * n);
    }
    
    zNode *zNodePt = zTail->previous;
    dcdwNode *dcdwNodePt = dcdwTail->previous;
    dcdbNode *dcdbNodePt = dcdbTail->previous;
    
    unsigned int l = nn->parameters->numberOfLayers-2;
    while (dcdwNodePt != NULL && dcdbNodePt != NULL) {
        aNodePt = aNodePt->previous;
        
        float sp[zNodePt->n];
        for (int i=0; i<zNodePt->n; i++) {
            sp[i] = sigmoidPrime(zNodePt->z[i]);
        }
        
        cblas_sgemv(CblasRowMajor, CblasTrans, (int)nn->weightsDimensions[l].m, (int)nn->weightsDimensions[l].n, 1.0, nn->weights+stride, (int)nn->weightsDimensions[l].n, delta, 1, 0.0, buffer, 1);
        for (int i=0; i<zNodePt->n; i++) {
            delta[i] = buffer[i] * sp[i];
        }
        // dc/dw at layer l
        for (int i=0; i<dcdwNodePt->m; i++) {
            for (int j=0; j<dcdwNodePt->n; j++) {
                dcdwNodePt->dcdw[i][j] = aNodePt->a[j]*delta[i];
            }
        }
        // dc/db at layer l
        for (int i=0; i<dcdbNodePt->n; i++) {
            dcdbNodePt->dcdb[i] = delta[i];
        }
        
        zNodePt = zNodePt->previous;
        dcdwNodePt = dcdwNodePt->previous;
        dcdbNodePt = dcdbNodePt->previous;
        stride = stride - (nn->weightsDimensions[l-1].m * nn->weightsDimensions[l-1].n);
        l--;
    }
    
    return NULL;
}


static int eval(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    float result = 0.0f;
    activationNode *aNodePt = NULL;
    
    int sum = 0;
    for (int k=0; k<nn->data->test->m; k++) {
        
        aNodePt = nn->activationsList;
        for (int i=0; i<nn->number_of_features; i++) {
            aNodePt->a[i] = nn->data->test->set[k][i];
        }

        nn->feedforward(self);
        
        aNodePt = nn->activationsList;
        while (aNodePt != NULL && aNodePt->next != NULL) {
            aNodePt = aNodePt->next;
        }
        
        result = (float)argmax(aNodePt->a, aNodePt->n);
        sum = sum + (result == nn->data->test->set[k][nn->number_of_features]);
    }
    
    return sum;
}

static int evaluate(void * _Nonnull self) {
    
    extern bool metal;
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    int sum = 0;
    double rt = 0.0;
    
#ifdef __APPLE__
    if (metal) {
        unsigned int weightsTableSize = 0;
        unsigned int biasesTableSize = 0;
        for (int l=0; l<nn->parameters->numberOfLayers-1; l++) {
            weightsTableSize = weightsTableSize + (nn->weightsDimensions[l].m * nn->weightsDimensions[l].n);
            biasesTableSize = biasesTableSize + nn->biasesDimensions[l].n;
        }
        
        nn->gpu->allocate_buffers((void *)nn);
        nn->gpu->prepare("feedforward");
        nn->gpu->format_data(nn->data->test->set, nn->data->test->m, nn->number_of_features);
        
        float result[nn->data->test->m];
        rt = realtime();
        
        nn->gpu->feedforward((void *)nn, result);
        float vector_sum = 0.0;
        vDSP_sve(result, 1, &vector_sum, nn->data->test->m);
        sum = (int)vector_sum;
        
        rt = realtime() - rt;
        
    } else {
        rt = realtime();
        sum = eval(self);
        rt = realtime() - rt;
    }
#else
    rt = realtime();
    sum = eval(self);
    rt = realtime() - rt;
#endif
    
    fprintf(stdout, "%s: total infer time in evaluation for %u input test data (s): %f\n", PROGRAM_NAME, nn->data->test->m, rt);
    
    return sum;
}

//
//  Compute the total cost function using a cross-entropy formulation
//
static float totalCost(void * _Nonnull self, float * _Nonnull * _Nonnull data, unsigned int m, bool convert) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    float norm, sum;
    activationNode *aNodePt = NULL;
    
    float cost = 0.0f;
    for (int i=0; i<m; i++) {

        aNodePt = nn->activationsList;
        for (int j=0; j<nn->number_of_features; j++) {
            aNodePt->a[j] = data[i][j];
        }
        
        nn->feedforward(self);
        aNodePt = nn->activationsList;
        while (aNodePt != NULL && aNodePt->next != NULL) {
            aNodePt = aNodePt->next;
        }
        
        float y[aNodePt->n];
        memset(y, 0.0f, sizeof(y));
        if (convert == true) {
            for (int j=0; j<aNodePt->n; j++) {
                if (data[i][nn->number_of_features] == nn->parameters->classifications[j]) {
                    y[j] = 1.0f;
                }
            }
        } else {
            int idx = (int)nn->number_of_features;
            for (int j=0; j<aNodePt->n; j++) {
                y[j] = data[i][idx];
                idx++;
            }
        }
        cost = cost + crossEntropyCost(aNodePt->a, y, aNodePt->n) / m;
        
        sum = 0.0f;
        unsigned int stride = 0;
        for (int l=0; l<nn->parameters->numberOfLayers-1; l++) {
            unsigned int m = nn->weightsDimensions[l].m;
            unsigned int n = nn->weightsDimensions[l].n;
            norm = frobeniusNorm(nn->weights+stride, (m * n));
            sum = sum + (norm*norm);
            stride = stride + (m * n);
        }
        cost = cost + 0.5f*(nn->parameters->lambda/(float)m)*sum;
    }
    
    return cost;
}

//
//  Return the output of the network for a given activation input
//
static void feedforward(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;

    activationNode *aNodePt = nn->activationsList;
    zNode *zNodePt = nn->zsList;
    
    unsigned int stride1 = 0;
    unsigned int stride2 = 0;
    for (int l=0; l<nn->parameters->numberOfLayers-1; l++) {
        unsigned int m = nn->weightsDimensions[l].m;
        unsigned int n = nn->weightsDimensions[l].n;
        
        aNodePt = aNodePt->next;
        zNodePt = zNodePt->next;
        float buffer[aNodePt->n];
        memset(buffer, 0.0f, sizeof(buffer));
        
        cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)m, (int)n, 1.0, nn->weights+stride1, (int)n, aNodePt->previous->a, 1, 0.0, buffer, 1);
#ifdef __APPLE__
        vDSP_vadd(buffer, 1, nn->biases+stride2, 1, zNodePt->z, 1,nn->biasesDimensions[l].n);
#else
        for (int i=0; i<nn->biasesDimensions[l].n; i++) {
            zNodePt->z[i] = buffer[i] + nn->biases[stride2+i];
        }
#endif
        for (int i=0; i<aNodePt->n; i++) {
            aNodePt->a[i] = sigmoid(zNodePt->z[i]);
        }
        nanToNum(aNodePt->a, aNodePt->n);
        
        stride1 = stride1 + (m * n);
        stride2 = stride2 + nn->biasesDimensions[l].n;
    }
}
