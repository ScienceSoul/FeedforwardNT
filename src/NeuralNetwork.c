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

static float * _Nonnull * _Nonnull createTrainigData(float * _Nonnull * _Nonnull dataSet, size_t start, size_t end, size_t * _Nonnull t1, size_t * _Nonnull t2, int * _Nonnull classifications, size_t numberOfClassifications, int * _Nonnull inoutSizes);

static float * _Nonnull * _Nonnull getData(float * _Nonnull * _Nonnull dataSet, size_t len1, size_t len2, size_t start, size_t end, size_t * _Nonnull t1, size_t * _Nonnull t2);

static weightNode * _Nonnull allocateWeightNode(void);
static biasNode * _Nonnull allocateBiasNode(void);

static activationNode * _Nonnull allocateActivationNode(void);
static zNode * _Nonnull allocateZNode(void);

static dcdwNode * _Nonnull allocateDcdwNode(void);
static dcdbNode * _Nonnull allocateDcdbNode(void);

static weightNode * _Nonnull initWeightsList(int * _Nonnull ntLayers, size_t numberOfLayers);
static biasNode * _Nonnull initBiasesList(int * _Nonnull ntLayers, size_t numberOfLayers);

static activationNode * _Nonnull initActivationsList(int * _Nonnull ntLayers, size_t numberOfLayers);
static zNode * _Nonnull initZsList(int * _Nonnull ntLayers, size_t numberOfLayers);

static dcdwNode * _Nonnull initDcdwList(int * _Nonnull ntLayers, size_t numberOfLayers);
static dcdbNode * _Nonnull initDcdbList(int * _Nonnull ntLayers, size_t numberOfLayers);

static void genesis(void * _Nonnull self);
static void finale(void * _Nonnull self);

static void computeNeural(void * _Nonnull self, bool * _Nullable showTotalCost);

static void miniBatch(void * _Nonnull self, float * _Nonnull * _Nonnull miniBatch);

static void updateWeightsBiases(void * _Nonnull self);

static void batchAccumulation(void * _Nonnull self);

static void * _Nullable backpropagation(void * _Nonnull self);

static int evaluate(void * _Nonnull self);

static float totalCost(void * _Nonnull self, float * _Nonnull * _Nonnull data, size_t m, bool convert);

static void feedforward(void * _Nonnull self);

#ifdef USE_OPENCL_GPU

static gpuInference * _Nonnull allocateGPUInference(void) {
    gpuInference *inference = (gpuInference *)malloc(sizeof(gpuInference));
    *inference = (gpuInference){.m=0, .n=0, .W=NULL, .A=NULL, .B=NULL, .Z=NULL, .kernel=NULL, .next=NULL, .previous=NULL};
    return inference;
}

static gpuInference * _Nonnull initGPUInferenceStore(GPUCompute *compute, weightNode * _Nonnull weightsList, activationNode * _Nonnull activationsList, int * _Nonnull ntLayers, size_t numberOfLayers) {
    
    cl_int err;
    
    weightNode *wNodePt = weightsList;
    activationNode *aNodePt = activationsList;
    gpuInference *inferenceList = allocateGPUInference();
    
    // The list head
    inferenceList->m = ntLayers[1];
    inferenceList->n = ntLayers[0];
    inferenceList->W = clCreateBuffer(compute->context, CL_MEM_READ_ONLY, (inferenceList->m*inferenceList->n)*sizeof(float),
                                  NULL, &err);
    if(err < 0) {
        fatal(PROGRAM_NAME, "problem creating GPU buffer for weights.");
    };
    
    inferenceList->A = clCreateBuffer(compute->context, CL_MEM_READ_ONLY, inferenceList->n*sizeof(float),
                                  NULL, &err);
    if(err < 0) {
        fatal(PROGRAM_NAME, "problem creating GPU buffer for activations.");
    };
    
    inferenceList->B = clCreateBuffer(compute->context, CL_MEM_READ_ONLY, inferenceList->m*sizeof(float),
                                  NULL, &err);
    if(err < 0) {
        fatal(PROGRAM_NAME, "problem creating GPU buffer for biases.");
    };
    
    inferenceList->Z = clCreateBuffer(compute->context, CL_MEM_WRITE_ONLY, inferenceList->m*sizeof(float),
                                  NULL, &err);
    if(err < 0) {
        fatal(PROGRAM_NAME, "problem creating GPU buffer for sgemv result vector.");
    };
    
    // The rest of the list
    int idx = 1;
    int k = 1;
    gpuInference *inferenceNodePt = inferenceList;
    while (k < numberOfLayers-1) {
        gpuInference *newNode = allocateGPUInference();
        wNodePt = wNodePt->next;
        aNodePt = aNodePt->next;
        newNode->m = ntLayers[idx+1];
        newNode->n = ntLayers[idx];
        newNode->W = clCreateBuffer(compute->context, CL_MEM_READ_ONLY, (newNode->m*newNode->n)*sizeof(float),
                                    NULL, &err);
        if(err < 0) {
            fatal(PROGRAM_NAME, "problem creating GPU buffer for weights.");
        };
        
        newNode->A = clCreateBuffer(compute->context, CL_MEM_READ_ONLY, newNode->n*sizeof(float),
                                    NULL, &err);
        if(err < 0) {
            fatal(PROGRAM_NAME, "problem creating GPU buffer for activations.");
        };
        
        newNode->B = clCreateBuffer(compute->context, CL_MEM_READ_ONLY, newNode->m*sizeof(float),
                                    NULL, &err);
        if(err < 0) {
            fatal(PROGRAM_NAME, "problem creating GPU buffer for biases");
        };
        
        newNode->Z = clCreateBuffer(compute->context, CL_MEM_WRITE_ONLY, newNode->m*sizeof(float),
                                    NULL, &err);
        if(err < 0) {
            fatal(PROGRAM_NAME, "problem creating GPU buffer for sgemv result vector.");
        };

        newNode->previous = inferenceNodePt;
        inferenceNodePt->next = newNode;
        inferenceNodePt = newNode;
        k++;
        idx++;
    }
    
    // Set up kernels and their arguments
    inferenceNodePt = inferenceList;
    while (inferenceNodePt != NULL) {
        inferenceNodePt->kernel = clCreateKernel(compute->program, "inference", &err);
        if (err < 0) {
            fatal(PROGRAM_NAME, "can't create kernel < sgemv >. Error: ", err);
        }
        
        err  = clSetKernelArg(inferenceNodePt->kernel, 0, sizeof(cl_int),   &inferenceNodePt->m);
        err |= clSetKernelArg(inferenceNodePt->kernel, 1, sizeof(cl_int),   &inferenceNodePt->n);
        err |= clSetKernelArg(inferenceNodePt->kernel, 2, sizeof(cl_mem),   &inferenceNodePt->W);
        err |= clSetKernelArg(inferenceNodePt->kernel, 3, sizeof(cl_mem),   &inferenceNodePt->A);
        err |= clSetKernelArg(inferenceNodePt->kernel, 4, sizeof(cl_mem),   &inferenceNodePt->B);
        err |= clSetKernelArg(inferenceNodePt->kernel, 5, sizeof(cl_mem),   &inferenceNodePt->Z);
        if(err < 0) {
            fatal(PROGRAM_NAME, "couldn't set an argument for the sgemv kernel.");
        };
        inferenceNodePt = inferenceNodePt->next;
    }
    
    return inferenceList;
}

static GPUCompute * _Nonnull  allocateGPUCompute(void) {
    
    GPUCompute *compute = (GPUCompute *)malloc(sizeof(GPUCompute));
    *compute = (GPUCompute){.gpuInferenceStore=NULL, .program=NULL, .device=NULL, .context=NULL, .queue=NULL};
    compute->inference = inference;
    return compute;
}

static void setUpOpenCLDevice(GPUCompute *compute) {
    
    cl_int err;
    char * _Nullable kernel_source;
    size_t src_len;
    
    compute->device = find_single_device();
    fprintf(stdout, "%s: GPU info: \n", PROGRAM_NAME);
    device_info(compute->device);
    
    compute->context = clCreateContext(0, 1, &compute->device, NULL, NULL, &err);
    if (err < 0) {
        fatal(PROGRAM_NAME, "can't create context for device. Error: ", err);
    }
    compute->queue = clCreateCommandQueue(compute->context, compute->device, CL_QUEUE_PROFILING_ENABLE, NULL);
    
    int success = LoadFileIntoString(OPENCL_PROGRAM_FILE_LOC1, &kernel_source, &src_len);
    if (success < 0) {
        success = LoadFileIntoString(OPENCL_PROGRAM_FILE_LOC2, &kernel_source, &src_len);
        if (success < 0) {
            fatal(PROGRAM_NAME, "can't load kernel source.");
        }
    }
    
    // Allocate program and kernel
    
    compute->program = clCreateProgramWithSource(compute->context, 1, (const char**)&kernel_source, NULL, &err);
    if (err < 0) {
        fatal(PROGRAM_NAME, "can't create program. Error: ", err);
    }
    

    const char *options = "-cl-mad-enable -cl-denorms-are-zero -cl-fast-relaxed-math";
    fprintf(stdout, "%s: build GPU program...\n", PROGRAM_NAME);
    err = clBuildProgram(compute->program, 1, &compute->device, options, NULL, &err);
    if (err < 0) {
        size_t log_size;
        clGetProgramBuildInfo(compute->program, compute->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *program_log = (char *)malloc(log_size+1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(compute->program, compute->device, CL_PROGRAM_BUILD_LOG, log_size+1, program_log, NULL);
        fprintf(stderr, "%s: error when buiding the GPU kernel.\n", PROGRAM_NAME);
        fprintf(stderr, "%s: log: %s\n", PROGRAM_NAME, program_log);
        free(program_log);
        fatal(PROGRAM_NAME);
    }
    fprintf(stdout, "%s: done.\n", PROGRAM_NAME);
}

static void inference(void * _Nonnull self, gpuInference * _Nonnull gpuInferenceStore) {
    
    cl_int err;
    
    GPUCompute *compute = (GPUCompute *)self;
    
    // Enqueue kernel
    size_t globalSize = gpuInferenceStore->m;
    err = clEnqueueNDRangeKernel(compute->queue, gpuInferenceStore->kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
#ifdef DEBUG
    if(err < 0) {
        fatal(PROGRAM_NAME, "couldn't enqueue the kernel.");
    }
#endif
    err = clFinish(compute->queue);
#ifdef DEBUG
    if (err < 0) {
        fatal(PROGRAM_NAME, "can't finish kernel. Error: ", err);
    }
#endif
}

#endif

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
            parseArgument(string, "inouts", nn->parameters->inoutSizes, &nn->parameters->numberOfInouts);
        }
        if (lineCount == 8) {
            nn->parameters->epochs = atoi(string);
        }
        if (lineCount == 9) {
            nn->parameters->miniBatchSize = atoi(string);
        }
        if (lineCount == 10) {
            nn->parameters->eta = strtof(string, NULL);
        }
        if (lineCount == 11) {
            nn->parameters->lambda = strtof(string, NULL);
        }
        lineCount++;
    }
    
    if (nn->parameters->numberOfDataDivisions != 2) {
        fatal(PROGRAM_NAME, " input data set should only be divided in two parts: one for training, one for testing.");
    }
    if (nn->parameters->numberOfInouts != 2) {
        fatal(PROGRAM_NAME, "only define one size for inputs and one size for outputs.");
    }
    if (nn->parameters->ntLayers[0] != nn->parameters->inoutSizes[0] || nn->parameters->ntLayers[(int)(nn->parameters->numberOfLayers)-1] != nn->parameters->inoutSizes[1]) {
        fatal(PROGRAM_NAME, "mismatch between size of network first/last layer and the nunmber of inputs/outputs.");
    }
    if (nn->parameters->inoutSizes[1] < nn->parameters->numberOfClassifications || nn->parameters->inoutSizes[1] > nn->parameters->numberOfClassifications) {
        fatal(PROGRAM_NAME, "mismatch between number of classifications and the number of outputs.");
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
    
    size_t len1=0, len2=0;
    float **raw_training = NULL;
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    fprintf(stdout, "%s: load the <%s> training data set.\n", PROGRAM_NAME, dataSetName);
    raw_training = nn->data->training->reader(trainFile, &len1, &len2);
    shuffle(raw_training, len1, len2);
    
    nn->data->training->set = createTrainigData(raw_training, 0, nn->parameters->dataDivisions[0], &nn->data->training->m, &nn->data->training->n, nn->parameters->classifications, nn->parameters->numberOfClassifications, nn->parameters->inoutSizes);
    
    if (testData) {
        nn->data->test->set = nn->data->test->reader(testFile, &nn->data->test->m, &nn->data->test->n);
        nn->data->validation->set = getData(raw_training, len1, len2, nn->parameters->dataDivisions[0], nn->parameters->dataDivisions[1], &nn->data->validation->m, &nn->data->validation->n);
    } else {
        nn->data->test->set = getData(raw_training, len1, len2, nn->parameters->dataDivisions[0], nn->parameters->dataDivisions[1], &nn->data->test->m, &nn->data->test->n);
    }
}

static float * _Nonnull * _Nonnull createTrainigData(float * _Nonnull * _Nonnull dataSet, size_t start, size_t end, size_t * _Nonnull t1, size_t * _Nonnull t2, int * _Nonnull classifications, size_t numberOfClassifications, int * _Nonnull inoutSizes) {
    
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
        
        // Binarization of the input ground-truth to get a one-hot-vector
        for (int k=0; k<numberOfClassifications; k++) {
            if (dataSet[i][inoutSizes[0]] == (float)classifications[k]) {
                trainingData[i][inoutSizes[0]+k] = 1.0f;
            } else trainingData[i][inoutSizes[0]+k] = 0.0f;
        }
    }
    
    return trainingData;
}

static float * _Nonnull * _Nonnull getData(float * _Nonnull * _Nonnull dataSet, size_t len1, size_t len2, size_t start, size_t end, size_t * _Nonnull t1, size_t * _Nonnull t2) {
    
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

static weightNode * _Nonnull allocateWeightNode(void) {
    
    weightNode *list = (weightNode *)malloc(sizeof(weightNode));
    *list = (weightNode){.m=0, .n=0, .w=NULL, .next=NULL, .previous=NULL};
    return list;
}

static biasNode * _Nonnull allocateBiasNode(void) {
    biasNode *list = (biasNode *)malloc(sizeof(biasNode));
    *list = (biasNode){.n=0, .b=NULL, .next=NULL, .previous=NULL};
    return list;
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
//  Create the weights list according to the number of layers in the network.
//  The weights are initialized using a Gaussian distribution with mean 0
//  and standard deviation 1 over the square root of the number of
//  weights connecting to the same neuron.
//
//  Return a pointer to the list head.
//
static weightNode * _Nonnull initWeightsList(int * _Nonnull ntLayers, size_t numberOfLayers) {
    
    weightNode *weightsList = allocateWeightNode();
    
    // The first weight node (i.e., layer)
    weightsList->w = floatmatrix(0, ntLayers[1]-1, 0, ntLayers[0]-1);
    weightsList->m = ntLayers[1];
    weightsList->n = ntLayers[0];
    // The rest of the weight nodes (i.e., layers)
    int idx = 1;
    int k = 1;
    weightNode *wNodePt = weightsList;
    while (k < numberOfLayers-1) {
        weightNode *newNode = allocateWeightNode();
        newNode->w = floatmatrix(0, ntLayers[idx+1]-1, 0, ntLayers[idx]-1);
        newNode->m = ntLayers[idx+1];
        newNode->n = ntLayers[idx];
        newNode->previous = wNodePt;
        wNodePt->next = newNode;
        wNodePt = newNode;
        k++;
        idx++;
    }
    
    wNodePt = weightsList;
    while (wNodePt != NULL) {
        for (int i = 0; i<wNodePt->m; i++) {
            for (int j=0; j<wNodePt->n; j++) {
                wNodePt->w[i][j] = randn(0.0f, 1.0f) / sqrtf((float)wNodePt->n);
            }
        }
        wNodePt = wNodePt->next;
    }
    
    return weightsList;
}

//
//  Create the biases list according to the number of layers in the network.
//  The biases are initialized using a Gaussian distribution with mean 0
//  and standard deviation 1.
//
//  Return a pointer to the list head.
//
static biasNode * _Nonnull initBiasesList(int * _Nonnull ntLayers, size_t numberOfLayers) {
    
    biasNode *biasesList = allocateBiasNode();
    
    // The first bias node (i.e., layer)
    biasesList->b = floatvec(0, ntLayers[1]-1);
    biasesList->n = ntLayers[1];
    // The rest of the bias nodes (i.e., layers)
    int idx = 2;
    int k = 1;
    biasNode *bNodePt = biasesList;
    while (k < numberOfLayers-1) {
        biasNode *newNode = allocateBiasNode();
        newNode->b = floatvec(0, ntLayers[idx]-1);
        newNode->n = ntLayers[idx];
        newNode->previous = bNodePt;
        bNodePt->next = newNode;
        bNodePt = newNode;
        k++;
        idx++;
    }
    
    bNodePt = biasesList;
    while (bNodePt != NULL) {
        for (int i = 0; i<bNodePt->n; i++) {
            bNodePt->b[i] = randn(0.0f, 1.0f);
        }
        bNodePt = bNodePt->next;
    }
    
    return biasesList;
}

//
//  Create the activations list according to the number of layers in the network.
//
//  Return a pointer to the list head.
//
static activationNode * _Nonnull initActivationsList(int * _Nonnull ntLayers, size_t numberOfLayers) {
    
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
static zNode * _Nonnull initZsList(int * _Nonnull ntLayers, size_t numberOfLayers) {
    
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
static dcdwNode * _Nonnull initDcdwList(int * _Nonnull ntLayers, size_t numberOfLayers) {
    
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
static dcdbNode * _Nonnull initDcdbList(int * _Nonnull ntLayers, size_t numberOfLayers) {
    
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
    *nn = (NeuralNetwork){.weightsList=NULL, .biasesList=NULL, .activationsList=NULL, .zsList=NULL,
                          .dcdwsList=NULL, .dcdbsList=NULL, .delta_dcdwsList=NULL, .delta_dcdbsList=NULL};
    
    nn->parameters = (parameters *)malloc(sizeof(parameters));
    nn->parameters->epochs = 0;
    nn->parameters->miniBatchSize = 0;
    nn->parameters->eta = 0.0f;
    nn->parameters->lambda = 0.0f;
    memset(nn->parameters->ntLayers, 0, sizeof(nn->parameters->ntLayers));
    memset(nn->parameters->classifications, 0, sizeof(nn->parameters->classifications));
    memset(nn->parameters->dataDivisions, 0, sizeof(nn->parameters->dataDivisions));
    memset(nn->parameters->inoutSizes, 0, sizeof(nn->parameters->inoutSizes));
    nn->load = loadParameters;
    
    nn->genesis = genesis;
    nn->finale = finale;
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
    
    nn->weightsList = initWeightsList(nn->parameters->ntLayers, nn->parameters->numberOfLayers);
    nn->biasesList = initBiasesList(nn->parameters->ntLayers, nn->parameters->numberOfLayers);
    nn->activationsList = initActivationsList(nn->parameters->ntLayers, nn->parameters->numberOfLayers);
    nn->zsList = initZsList(nn->parameters->ntLayers, nn->parameters->numberOfLayers);
    nn->dcdwsList = initDcdwList(nn->parameters->ntLayers, nn->parameters->numberOfLayers);
    nn->dcdbsList = initDcdbList(nn->parameters->ntLayers, nn->parameters->numberOfLayers);
    nn->delta_dcdwsList = initDcdwList(nn->parameters->ntLayers, nn->parameters->numberOfLayers);
    nn->delta_dcdbsList = initDcdbList(nn->parameters->ntLayers, nn->parameters->numberOfLayers);
    
#ifdef USE_OPENCL_GPU
    // Allocate the GPU compute environment
    nn->compute = allocateGPUCompute();
    setUpOpenCLDevice(nn->compute);
    nn->compute->gpuInferenceStore = initGPUInferenceStore(nn->compute, nn->weightsList, nn->activationsList, ntLayers, numberOfLayers);
#endif
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
    
    weightNode *wTail = nn->weightsList;
    while (wTail != NULL && wTail->next != NULL) {
        wTail = wTail->next;
    }
    weightNode *wNodePt = NULL;
    while (wTail != NULL) {
        wNodePt = wTail->previous;
        free_fmatrix(wTail->w, 0, wTail->m-1, 0, wTail->n-1);
        wTail->w = NULL;
        wTail->next = NULL;
        wTail->previous = NULL;
        free(wTail);
        wTail = wNodePt;
    }
    
    biasNode *bTail = nn->biasesList;
    while (bTail != NULL && bTail->next != NULL) {
        bTail = bTail->next;
    }
    biasNode *bNodePt = NULL;
    while (bTail != NULL) {
        bNodePt = bTail->previous;
        free_fvector(bTail->b, 0, bTail->n);
        bTail->b = NULL;
        bTail->next = NULL;
        bTail->previous = NULL;
        free(bTail);
        bTail = bNodePt;
    }
    
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
    
#ifdef USE_OPENCL_GPU
    gpuInference *inferenceTail = nn->compute->gpuInferenceStore;
    while (inferenceTail != NULL && inferenceTail->next != NULL) {
        inferenceTail = inferenceTail->next;
    }
    gpuInference *inferenceNodePt = NULL;
    while (inferenceTail != NULL) {
        inferenceNodePt = inferenceTail->previous;
        if (inferenceTail->W != NULL) {
            clReleaseMemObject(inferenceTail->W);
            clReleaseMemObject(inferenceTail->A);
            clReleaseMemObject(inferenceTail->B);
            clReleaseMemObject(inferenceTail->Z);
            clReleaseKernel(inferenceTail->kernel);
        }
        inferenceTail->next = NULL;
        inferenceTail->previous = NULL;
        free(inferenceTail);
        inferenceTail = inferenceNodePt;
    }
    
    clReleaseProgram(nn->compute->program);
    clReleaseContext(nn->compute->context);
    clReleaseCommandQueue(nn->compute->queue);
    free(nn->compute);
#endif
}

static void computeNeural(void * _Nonnull self, bool * _Nullable showTotalCost) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    nn->number_of_features = nn->parameters->inoutSizes[0];
    
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
            fprintf(stdout, "%s: Epoch {%d/%d}: testing network with {%zu} inputs:\n", PROGRAM_NAME, k, nn->parameters->epochs, nn->data->test->m);
            int result = nn->evaluate(self);
            fprintf(stdout, "%s: Epoch {%d/%d}: {%d} / {%zu}.\n", PROGRAM_NAME, k, nn->parameters->epochs, result, nn->data->test->m);
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
    weightNode *wNodePt = nn->weightsList;
    dcdwNode *dcdwNodePt = nn->dcdwsList;
    while (wNodePt != NULL) {
        for (int i=0; i<wNodePt->m; i++) {
            for (int j=0; j<wNodePt->n; j++) {
                wNodePt->w[i][j] = (1.0f-((nn->parameters->eta*nn->parameters->lambda)/(float)nn->data->training->m))*wNodePt->w[i][j] - (nn->parameters->eta/(float)nn->parameters->miniBatchSize)*dcdwNodePt->dcdw[i][j];
            }
        }
        wNodePt = wNodePt->next;
        dcdwNodePt = dcdwNodePt->next;
    }
    
    // Update biases
    biasNode *bNodePt = nn->biasesList;
    dcdbNode *dcdbNodePt = nn->dcdbsList;
    while (bNodePt != NULL) {
        for (int i=0; i<bNodePt->n; i++) {
            bNodePt->b[i] = bNodePt->b[i] - (nn->parameters->eta/(float)nn->parameters->miniBatchSize)*dcdbNodePt->dcdb[i];
        }
        bNodePt = bNodePt->next;
        dcdbNodePt = dcdbNodePt->next;
    }

// Update the weights and biases buffers in the GPU memory
#ifdef USE_OPENCL_GPU
    cl_int err;
    
    gpuInference *inferenceNodePt = nn->compute->gpuInferenceStore;
    wNodePt = nn->weightsList;
    bNodePt = nn->biasesList;
    while (inferenceNodePt != NULL) {
        if (inferenceNodePt->W != NULL) {
            void *_mapped_W = clEnqueueMapBuffer(nn->compute->queue, inferenceNodePt->W, CL_TRUE, CL_MAP_WRITE, 0, (inferenceNodePt->m*inferenceNodePt->n)*sizeof(float), 0, NULL, NULL, &err);
            if (err < 0) {
                fatal(PROGRAM_NAME, "couldn't map weights buffer from GPU.");
            }
            float *buffer = _mapped_W;
            memcpy(buffer, *wNodePt->w, (wNodePt->m*wNodePt->n)*sizeof(float));
            clEnqueueUnmapMemObject(nn->compute->queue, inferenceNodePt->W, _mapped_W, 0, NULL, NULL);
        }
        if (inferenceNodePt->B != NULL) {
            void *_mapped_B = clEnqueueMapBuffer(nn->compute->queue, inferenceNodePt->B, CL_TRUE, CL_MAP_WRITE, 0, inferenceNodePt->m*sizeof(float), 0, NULL, NULL, &err);
            if (err < 0) {
                fatal(PROGRAM_NAME, "couldn't map biases buffer from GPU.");
            }
            float *buffer = _mapped_B;
            memcpy(buffer, bNodePt->b, bNodePt->n*sizeof(float));
            clEnqueueUnmapMemObject(nn->compute->queue, inferenceNodePt->B, _mapped_B, 0, NULL, NULL);
        }
        inferenceNodePt = inferenceNodePt->next;
        wNodePt = wNodePt->next;
        bNodePt = bNodePt->next;
    }
#endif
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
    
    // Weights at last layer
    weightNode *wTail = nn->weightsList;
    while (wTail != NULL && wTail->next != NULL) {
        wTail = wTail->next;
    }
    
    weightNode *wNodePt = wTail;
    zNode *zNodePt = zTail->previous;
    dcdwNode *dcdwNodePt = dcdwTail->previous;
    dcdbNode *dcdbNodePt = dcdbTail->previous;
    
    while (dcdwNodePt != NULL && dcdbNodePt != NULL) {
        aNodePt = aNodePt->previous;
        
        float sp[zNodePt->n];
        for (int i=0; i<zNodePt->n; i++) {
            sp[i] = sigmoidPrime(zNodePt->z[i]);
        }
        
        cblas_sgemv(CblasRowMajor, CblasTrans, (int)wNodePt->m, (int)wNodePt->n, 1.0, *wNodePt->w, (int)wNodePt->n, delta, 1, 0.0, buffer, 1);
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
        
        wNodePt = wNodePt->previous;
        zNodePt = zNodePt->previous;
        dcdwNodePt = dcdwNodePt->previous;
        dcdbNodePt = dcdbNodePt->previous;
    }
    
    return NULL;
}

static int evaluate(void * _Nonnull self) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    float results = 0.0f;
    activationNode *aNodePt = NULL;
    
    int sum = 0;
    for (int k=0; k<nn->data->test->m; k++) {
#ifdef USE_OPENCL_GPU
        cl_int err;
        gpuInference *inferenceNodePt = nn->compute->gpuInferenceStore;
        void *_mapped_A = clEnqueueMapBuffer(nn->compute->queue, inferenceNodePt->A, CL_TRUE, CL_MAP_WRITE, 0, inferenceNodePt->n*sizeof(float), 0, NULL, NULL, &err);
        if (err < 0) {
            fatal(PROGRAM_NAME, "couldn't map result buffer from GPU.");
        }
        float *buffer = _mapped_A;
        for (int i=0; i<nn->number_of_features; i++) {
            buffer[i] = testData[k][i];
        }
        clEnqueueUnmapMemObject(nn->compute->queue, inferenceNodePt->A, _mapped_A, 0, NULL, NULL);
#else
        aNodePt = nn->activationsList;
        for (int i=0; i<nn->number_of_features; i++) {
            aNodePt->a[i] = nn->data->test->set[k][i];
        }
#endif
        double rt = realtime();
        nn->feedforward(self);
        rt = realtime() -  rt;
#ifdef VERBOSE
        fprintf(stdout, "%s: time to infer in evaluation (s): %f\n", PROGRAM_NAME, rt);
#endif
        aNodePt = nn->activationsList;
        while (aNodePt != NULL && aNodePt->next != NULL) {
            aNodePt = aNodePt->next;
        }
#ifdef USE_OPENCL_GPU
        inferenceNodePt = nn->compute->gpuInferenceStore;
        while (inferenceNodePt != NULL && inferenceNodePt->next != NULL) {
            inferenceNodePt = inferenceNodePt->next;
        }
        void *_mapped_Z = clEnqueueMapBuffer(nn->compute->queue, inferenceNodePt->Z, CL_TRUE, CL_MAP_READ, 0, inferenceNodePt->m*sizeof(float), 0, NULL, NULL, &err);
        if (err < 0) {
            fatal(PROGRAM_NAME, "couldn't map result buffer from GPU.");
        }
        buffer = _mapped_Z;
        memcpy(aNodePt->a, buffer, aNodePt->n*sizeof(float));
        clEnqueueUnmapMemObject(nn->compute->queue, inferenceNodePt->Z, _mapped_Z, 0, NULL, NULL);

#endif
        results = (float)argmax(aNodePt->a, aNodePt->n);
        sum = sum + (results == nn->data->test->set[k][nn->number_of_features]);
    }
    
    return sum;
}

//
//  Compute the total cost function using a cross-entropy formulation
//
static float totalCost(void * _Nonnull self, float * _Nonnull * _Nonnull data, size_t m, bool convert) {
    
    NeuralNetwork *nn = (NeuralNetwork *)self;
    
    float norm, sum;
    activationNode *aNodePt = NULL;
    
    float cost = 0.0f;
    for (int i=0; i<m; i++) {
#ifdef USE_OPENCL_GPU
        cl_int err;
        gpuInference *inferenceNodePt = nn->compute->gpuInferenceStore;
        void *_mapped_A = clEnqueueMapBuffer(nn->compute->queue, inferenceNodePt->A, CL_TRUE, CL_MAP_WRITE, 0, inferenceNodePt->n*sizeof(float), 0, NULL, NULL, &err);
        if (err < 0) {
            fatal(PROGRAM_NAME, "couldn't map result buffer from GPU.");
        }
        float *buffer = _mapped_A;
        for (int j=0; j<nn->number_of_features; j++) {
            buffer[j] = data[i][j];
        }
        clEnqueueUnmapMemObject(nn->compute->queue, inferenceNodePt->A, _mapped_A, 0, NULL, NULL);
#else
        aNodePt = nn->activationsList;
        for (int j=0; j<nn->number_of_features; j++) {
            aNodePt->a[j] = data[i][j];
        }
#endif
        nn->feedforward(self);
        aNodePt = nn->activationsList;
        while (aNodePt != NULL && aNodePt->next != NULL) {
            aNodePt = aNodePt->next;
        }
#ifdef USE_OPENCL_GPU
        inferenceNodePt = nn->compute->gpuInferenceStore;
        while (inferenceNodePt != NULL && inferenceNodePt->next != NULL) {
            inferenceNodePt = inferenceNodePt->next;
        }
        void *_mapped_Z = clEnqueueMapBuffer(nn->compute->queue, inferenceNodePt->Z, CL_TRUE, CL_MAP_READ, 0, inferenceNodePt->m*sizeof(float), 0, NULL, NULL, &err);
        if (err < 0) {
            fatal(PROGRAM_NAME, "couldn't map result buffer from GPU.");
        }
        buffer = _mapped_Z;
        memcpy(aNodePt->a, buffer, aNodePt->n*sizeof(float));
        clEnqueueUnmapMemObject(nn->compute->queue, inferenceNodePt->Z, _mapped_Z, 0, NULL, NULL);
        
#endif
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
        weightNode *wNodePt = nn->weightsList;
        while (wNodePt != NULL) {
            norm = frobeniusNorm(wNodePt->w, wNodePt->m, wNodePt->n);
            sum = sum + (norm*norm);
            wNodePt = wNodePt->next;
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
    
#ifdef USE_OPENCL_GPU
    cl_int err;
    
    gpuInference *inferenceNodePt = nn->compute->gpuInferenceStore;
    while (inferenceNodePt != NULL) {
        nn->compute->inference((void *)nn->compute, inferenceNodePt);
        if (inferenceNodePt->next != NULL) {
            err = clEnqueueCopyBuffer(nn->compute->queue, inferenceNodePt->Z, inferenceNodePt->next->A, 0, 0, inferenceNodePt->next->n*sizeof(float), 0, NULL, NULL);
    #ifdef DEBUG
            if(err < 0) {
                fatal(PROGRAM_NAME, "couldn't enqueue the buffer copy.");
            }
    #endif
        }
        inferenceNodePt = inferenceNodePt->next;
    }
#else
    weightNode *wNodePt = nn->weightsList;
    biasNode *bNodePt = nn->biasesList;
    activationNode *aNodePt = nn->activationsList;
    zNode *zNodePt = nn->zsList;
    
    while (wNodePt != NULL && bNodePt != NULL) {
        aNodePt = aNodePt->next;
        zNodePt = zNodePt->next;
        float buffer[aNodePt->n];
        memset(buffer, 0.0f, sizeof(buffer));
        
        cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)wNodePt->m, (int)wNodePt->n, 1.0, *wNodePt->w, (int)wNodePt->n, aNodePt->previous->a, 1, 0.0, buffer, 1);
    #ifdef __APPLE__
        vDSP_vadd(buffer, 1, bNodePt->b, 1, zNodePt->z, 1, bNodePt->n);
    #else
        for (int i=0; i<bNodePt->n; i++) {
            zNodePt->z[i] = buffer[i] + bNodePt->b[i];
        }
    #endif
        for (int i=0; i<aNodePt->n; i++) {
            aNodePt->a[i] = sigmoid(zNodePt->z[i]);
        }
        nanToNum(aNodePt->a, aNodePt->n);
        wNodePt = wNodePt->next;
        bNodePt = bNodePt->next;
    }
#endif
}
