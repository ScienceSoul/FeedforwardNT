//
//  Training.c
//  FeedforwardNT
//
//  Created by Seddik hakime on 21/05/2017.
//

#include <Accelerate/Accelerate.h>
#include "Training.h"
#include "TimeProfile.h"

weightNode * __nonnull allocateWeightNode(void) {
    
    weightNode *list = (weightNode *)malloc(sizeof(weightNode));
    *list = (weightNode){.m=0, .n=0, .w=NULL, .next=NULL, .previous=NULL};
    return list;
}

biasNode * __nonnull allocateBiasNode(void) {
    biasNode *list = (biasNode *)malloc(sizeof(biasNode));
    *list = (biasNode){.n=0, .b=NULL, .next=NULL, .previous=NULL};
    return list;
}

activationNode * __nonnull allocateActivationNode(void) {
    activationNode *list = (activationNode *)malloc(sizeof(activationNode));
    *list = (activationNode){.n=0, .a=NULL, .next=NULL, .previous=NULL};
    return list;
}

zNode * __nonnull allocateZNode(void) {
    zNode *list = (zNode *)malloc(sizeof(zNode));
    *list = (zNode){.n=0, .z=NULL, .next=NULL, .previous=NULL};
    return list;
}

dcdwNode * __nonnull allocateDcdwNode(void) {
    dcdwNode *list = (dcdwNode *)malloc(sizeof(dcdwNode));
    *list = (dcdwNode){.m=0, .n=0, .dcdw=NULL, .next=NULL, .previous=NULL};
    return list;
}

dcdbNode * __nonnull allocateDcdbNode(void) {
    dcdbNode *list = (dcdbNode *)malloc(sizeof(dcdbNode));
    *list = (dcdbNode){.n=0, .dcdb=NULL, .next=NULL, .previous=NULL};
    return list;
}

//
//  Allocate a single node in the batch
//
pthreadBatchNode * __nonnull allocatePthreadBatchNode(void) {
    
    pthreadBatchNode *node = (pthreadBatchNode *)malloc(sizeof(pthreadBatchNode));
    *node = (pthreadBatchNode){.index=0, .max=0, .batch=NULL, .weightsList=NULL, .biasesList=NULL, .activationsList=NULL, .zsList=NULL,
        .dcdwsList=NULL, .dcdbsList=NULL, .inoutSizes=NULL};
    return node;
}

void deallocate(weightNode * __nonnull weightsList, biasNode * __nonnull biasesList, dcdwNode * __nonnull dcdwsList, dcdbNode * __nonnull dcdbsList, activationNode * __nonnull activationsList, zNode * __nonnull zsList, pthreadBatchNode * __nullable * __nullable threadDataPt, pthread_t __nullable * __nullable  threadTID, int miniBatchSize, bool pthread) {
    
    if (pthread) {
        for (int i=0; i<miniBatchSize; i++) {
            pthreadBatchNode *node = threadDataPt[i];
            node->weightsList = NULL;
            node->biasesList = NULL;
            node->inoutSizes = NULL;
            
            activationNode *aTail = node->activationsList;
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
            
            zNode *zTail = node->zsList;
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
            
            dcdwNode *dcdwTail = node->dcdwsList;
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
            
            dcdbNode *dcdbTail = node->dcdbsList;
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
            free(node);
        }
        free(threadDataPt);
        free(threadTID);
    } else {
        pthreadBatchNode *node = threadDataPt[0];
        node->weightsList = NULL;
        node->biasesList = NULL;
        node->inoutSizes = NULL;
        node->activationsList = NULL;
        node->zsList = NULL;
        
        dcdwNode *dcdwTail = node->dcdwsList;
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
        
        dcdbNode *dcdbTail = node->dcdbsList;
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
        free(node);
        free(threadDataPt);
    }
    
    weightNode *wTail = weightsList;
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
    
    biasNode *bTail = biasesList;
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
    
    dcdwNode *dcdwTail = dcdwsList;
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
    
    dcdbNode *dcdbTail = dcdbsList;
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
    
    activationNode *aTail = activationsList;
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
    
    zNode *zTail = zsList;
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
}

//
//  Create the weights list according to the number of layers in the network.
//  The weights are initialized using a Gaussian distribution with mean 0
//  and standard deviation 1 over the square root of the number of
//  weights connecting to the same neuron.
//
//  Return a pointer to the list head.
//
weightNode * __nonnull initWeightsList(int * __nonnull ntLayers, size_t numberOfLayers) {
    
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
biasNode * __nonnull initBiasesList(int * __nonnull ntLayers, size_t numberOfLayers) {
    
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
activationNode * __nonnull initActivationsList(int * __nonnull ntLayers, size_t numberOfLayers) {
    
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
zNode * __nonnull initZsList(int * __nonnull ntLayers, size_t numberOfLayers) {
    
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
dcdwNode * __nonnull initDcdwList(int * __nonnull ntLayers, size_t numberOfLayers) {
    
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
dcdbNode * __nonnull initDcdbList(int * __nonnull ntLayers, size_t numberOfLayers) {
 
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

//  The sigmoid fonction
float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

// Derivative of the sigmoid function
float sigmoidPrime(float z) {
    return sigmoid(z) * (1.0f - sigmoid(z));
}

//
//  Return the output of the network for a given activation input
//
void feedforward(weightNode * __nonnull weights, activationNode * __nonnull activations, biasNode * __nonnull biases, zNode * __nonnull zs) {
    
    weightNode *wNodePt = weights;
    biasNode *bNodePt = biases;
    activationNode *aNodePt = activations;
    zNode *zNodePt = zs;
    
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
}

void updateMiniBatch(float * __nonnull * __nonnull miniBatch,
                     int miniBatchSize,
                     int * __nonnull inoutSizes,
                     weightNode * __nonnull weightsList,
                     biasNode * __nonnull biasesList,
                     dcdwNode * __nullable dcdwsList,
                     dcdbNode * __nullable dcdbsList,
                     pthreadBatchNode * __nullable * __nullable threadDataPt,
                     pthread_t __nullable * __nullable  threadTID,
                     int * __nonnull ntLayers,
                     size_t numberOfLayers, size_t tr1, float eta, float lambda, bool * __nullable pthread) {
    
    bool multiThreadedBatch = false;
    if (pthread != NULL) {
        multiThreadedBatch = (*pthread == true) ? true : false;
    }
    
    dcdwNode *dcdwNodePt = dcdwsList;
    while (dcdwNodePt != NULL) {
        memset(*dcdwNodePt->dcdw, 0.0f, (dcdwNodePt->m*dcdwNodePt->n)*sizeof(float));
        dcdwNodePt = dcdwNodePt->next;
    }
    
    dcdbNode *dcdbNodePt = dcdbsList;
    while (dcdbNodePt != NULL) {
        memset(dcdbNodePt->dcdb, 0.0f, dcdbNodePt->n*sizeof(float));
        dcdbNodePt = dcdbNodePt->next;
    }
    
    double rt = realtime();
    for (int i=0; i<miniBatchSize; i++) {
        if (multiThreadedBatch) {
            pthreadBatchNode *node = threadDataPt[i];
            node->index = i;
            node->batch = miniBatch;
            pthread_create(&(threadTID[i]), NULL, backpropagation, (void *)node);
        } else {
            pthreadBatchNode *node = threadDataPt[0];
            node->index = i;
            node->batch = miniBatch;
            backpropagation((void *)node);
            accumulateFromThreads(dcdwsList, dcdbsList, threadDataPt, miniBatchSize, multiThreadedBatch);
        }
    }
    
    if (multiThreadedBatch) {
        for (int i=0; i<miniBatchSize; i++) {
             pthread_join(threadTID[i], NULL);
        }
        accumulateFromThreads(dcdwsList, dcdbsList, threadDataPt, miniBatchSize, multiThreadedBatch);
    }
    rt = realtime() - rt;
    fprintf(stdout, "FeedforwardNT: time to complete a single batch (s): %f\n", rt);

    
    updateWeightsBiases(weightsList, biasesList, dcdwsList, dcdbsList, miniBatchSize, tr1, eta, lambda);
}

void updateWeightsBiases(weightNode * __nonnull weightsList, biasNode * __nonnull biasesList, dcdwNode * __nonnull dcdwsList, dcdbNode * __nonnull dcdbsList, int miniBatchSize, size_t tr1, float eta, float lambda) {
 
    // Update weights
    weightNode *wNodePt = weightsList;
    dcdwNode *dcdwNodePt = dcdwsList;
    while (wNodePt != NULL) {
        for (int i=0; i<wNodePt->m; i++) {
            for (int j=0; j<wNodePt->n; j++) {
                wNodePt->w[i][j] = (1.0f-eta*(lambda/tr1))*wNodePt->w[i][j] - (eta/miniBatchSize)*dcdwNodePt->dcdw[i][j];
            }
        }
        wNodePt = wNodePt->next;
        dcdwNodePt = dcdwNodePt->next;
    }
    
    // Update biases
    biasNode *bNodePt = biasesList;
    dcdbNode *dcdbNodePt = dcdbsList;
    while (bNodePt != NULL) {
        for (int i=0; i<bNodePt->n; i++) {
            bNodePt->b[i] = bNodePt->b[i] - (eta/miniBatchSize)*dcdbNodePt->dcdb[i];
        }
        bNodePt = bNodePt->next;
        dcdbNodePt = dcdbNodePt->next;
    }
}

void accumulateFromThreads(dcdwNode * __nonnull dcdwsList, dcdbNode * __nonnull dcdbsList, pthreadBatchNode * __nullable * __nullable threadDataPt, int miniBatchSize, bool pthread) {
    
    // Accumulate dcdw and dc/db from all threads if multithreaded or
    // from a single one if serial
    
    if (pthread) {
        for (int i=0; i<miniBatchSize; i++) {
            dcdwNode *dcdwNodePt = dcdwsList;
            dcdbNode *dcdbNodePt = dcdbsList;
            pthreadBatchNode *node = threadDataPt[i];
            dcdwNode *pthead_dcdwsPt = node->dcdwsList;
            dcdbNode *pthead_dcdbsPt = node->dcdbsList;
            while (dcdwNodePt != NULL && pthead_dcdwsPt != NULL) {
                for (int j=0; j<dcdwNodePt->m; j++) {
                    for (int k=0; k<dcdwNodePt->n; k++) {
                        dcdwNodePt->dcdw[j][k] = dcdwNodePt->dcdw[j][k] + pthead_dcdwsPt->dcdw[j][k];
                    }
                }
                for (int j=0; j<dcdbNodePt->n; j++) {
                    dcdbNodePt->dcdb[j] = dcdbNodePt->dcdb[j] + pthead_dcdbsPt->dcdb[j];
                }
                dcdwNodePt = dcdwNodePt->next;
                dcdbNodePt = dcdbNodePt->next;
                pthead_dcdwsPt = pthead_dcdwsPt->next;
                pthead_dcdbsPt = pthead_dcdbsPt->next;
            }
        }
    } else {
        dcdwNode *dcdwNodePt = dcdwsList;
        dcdbNode *dcdbNodePt = dcdbsList;
        pthreadBatchNode *node = threadDataPt[0];
        dcdwNode *pthead_dcdwsPt = node->dcdwsList;
        dcdbNode *pthead_dcdbsPt = node->dcdbsList;
        while (dcdwNodePt != NULL && pthead_dcdwsPt != NULL) {
            for (int j=0; j<dcdwNodePt->m; j++) {
                for (int k=0; k<dcdwNodePt->n; k++) {
                    dcdwNodePt->dcdw[j][k] = dcdwNodePt->dcdw[j][k] + pthead_dcdwsPt->dcdw[j][k];
                }
            }
            for (int j=0; j<dcdbNodePt->n; j++) {
                dcdbNodePt->dcdb[j] = dcdbNodePt->dcdb[j] + pthead_dcdbsPt->dcdb[j];
            }
            dcdwNodePt = dcdwNodePt->next;
            dcdbNodePt = dcdbNodePt->next;
            pthead_dcdwsPt = pthead_dcdwsPt->next;
            pthead_dcdbsPt = pthead_dcdbsPt->next;
        }
    }
}

//
//  Return the gradient of the cross-entropy cost function C_x layers by layers
//
void * __nullable backpropagation(void * __nonnull node) {
    
    pthreadBatchNode *entry = (pthreadBatchNode *)node;
    
    // Activations at the input layer
    activationNode *aNodePt = entry->activationsList;
    for (int i=0; i<entry->inoutSizes[0]; i++) {
        aNodePt->a[i] = entry->batch[entry->index][i];
    }
    
    // Feedforward
    feedforward(entry->weightsList, entry->activationsList, entry->biasesList, entry->zsList);

    // ------------- Backward pass
    // At last layer
    
    activationNode *aTail = entry->activationsList;
    while (aTail != NULL && aTail->next != NULL) {
        aTail = aTail->next;
    }
    zNode *zTail = entry->zsList;
    while (zTail != NULL && zTail->next != NULL) {
        zTail = zTail->next;
    }
    
    float delta[entry->max];
    float buffer[entry->max];
    
    // Compute delta
    int k = entry->inoutSizes[0];
    for (int i=0; i<aTail->n; i++) {
        delta[i] = aTail->a[i] - entry->batch[entry->index][k];
        k++;
    }

    //dc/dw and dc/db at last layer
    dcdwNode *dcdwTail = entry->dcdwsList;
    while (dcdwTail != NULL && dcdwTail->next != NULL) {
        dcdwTail = dcdwTail->next;
    }
    dcdbNode *dcdbTail = entry->dcdbsList;
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
    weightNode *wTail = entry->weightsList;
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


int evaluate(float * __nonnull * __nonnull testData, size_t ts1, int * __nonnull inoutSizes, weightNode * __nonnull weightsList, biasNode * __nonnull biasesList, activationNode * __nonnull activationsList, zNode * __nonnull zsList) {
    
    float results[ts1];
    activationNode *aNodePt = NULL;
    
    int sum = 0;
    for (int k=0; k<ts1; k++) {
        aNodePt = activationsList;
        for (int i=0; i<inoutSizes[0]; i++) {
            aNodePt->a[i] = testData[k][i];
        }
        feedforward(weightsList, activationsList, biasesList, zsList);
        aNodePt = activationsList;
        while (aNodePt != NULL && aNodePt->next != NULL) {
            aNodePt = aNodePt->next;
        }
        results[k] = (float)argmax(aNodePt->a, aNodePt->n);
        sum = sum + (results[k] == testData[k][inoutSizes[0]]);
    }

    return sum;
}

//
//  Compute the Frobenius norm of a m x n matrix
//
float frobeniusNorm(float * __nonnull * __nonnull mat, size_t m, size_t n) {
    
    float norm = 0.0f;
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            norm = norm + powf(mat[i][j], 2.0f);
        }
    }
    
    return sqrtf(norm);
}

float crossEntropyCost(float * __nonnull a, float * __nonnull y, size_t n) {
    
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

//
//  Compute the total cost function using a cross-entropy formulation
//
float totalCost(float * __nonnull * __nonnull data, size_t m, weightNode * __nonnull weightsList, biasNode * __nonnull biasesList, activationNode * __nonnull activationsList, zNode * __nonnull zsList, int * __nonnull inoutSizes, int * __nonnull classifications, float lambda, bool convert) {
    
    float norm, sum;
    activationNode *aNodePt = NULL;
    
    float cost = 0.0f;
    for (int i=0; i<m; i++) {
        aNodePt = activationsList;
        for (int j=0; j<inoutSizes[0]; j++) {
            aNodePt->a[j] = data[i][j];
        }
        feedforward(weightsList, activationsList, biasesList, zsList);
        aNodePt = activationsList;
        while (aNodePt != NULL && aNodePt->next != NULL) {
            aNodePt = aNodePt->next;
        }
        float y[aNodePt->n];
        memset(y, 0.0f, sizeof(y));
        if (convert == true) {
            for (int j=0; j<aNodePt->n; j++) {
                if (data[i][inoutSizes[0]] == classifications[j]) {
                    y[j] = 1.0f;
                }
            }
        } else {
            int idx = inoutSizes[0];
            for (int j=0; j<aNodePt->n; j++) {
                y[j] = data[i][idx];
                idx++;
            }
        }
        cost = cost + crossEntropyCost(aNodePt->a, y, aNodePt->n) / m;
        
        sum = 0.0f;
        weightNode *wNodePt = weightsList;
        while (wNodePt != NULL) {
            norm = frobeniusNorm(wNodePt->w, wNodePt->m, wNodePt->n);
            sum = sum + (norm*norm);
            wNodePt = wNodePt->next;
        }
        cost = cost + 0.5f*(lambda/m)*sum;
    }
    
    return cost;
}

void SDG(float * __nonnull * __nonnull trainingData, float * __nonnull * __nonnull testData, size_t tr1, size_t tr2, size_t ts1, size_t ts2, int * __nonnull ntLayers, size_t numberOfLayers, int * __nonnull inoutSizes, int * __nonnull classifications, int epochs, int miniBatchSize, float eta, float lambda, bool pthread) {
    
    pthreadBatchNode **threadDataPt = NULL;
    pthread_t *threadTID = NULL;

    // Allocate memory for the weight, bias, activation, z, dC/dx and dC/db data structures
    
    weightNode *weightsList = initWeightsList(ntLayers, numberOfLayers);
    biasNode *biasesList = initBiasesList(ntLayers, numberOfLayers);
    activationNode *activationsList = initActivationsList(ntLayers, numberOfLayers);
    zNode *zsList = initZsList(ntLayers, numberOfLayers);
    dcdwNode *dcdwsList = initDcdwList(ntLayers, numberOfLayers);
    dcdbNode *dcdbsList = initDcdbList(ntLayers, numberOfLayers);
    
    if (pthread) {
        threadDataPt = (pthreadBatchNode **)malloc(miniBatchSize * sizeof(pthreadBatchNode *));
        threadTID = (pthread_t *)malloc(miniBatchSize * sizeof(pthread_t));
        
        for (int i=0; i<miniBatchSize; i++) {
            pthreadBatchNode *node = allocatePthreadBatchNode();
            node->max = max_array(ntLayers, numberOfLayers);
            node->weightsList = weightsList;
            node->biasesList = biasesList;
            node->activationsList = initActivationsList(ntLayers, numberOfLayers);
            node->zsList = initZsList(ntLayers, numberOfLayers);
            node->dcdwsList = initDcdwList(ntLayers, numberOfLayers);
            node->dcdbsList = initDcdbList(ntLayers, numberOfLayers);
            node->inoutSizes = inoutSizes;
            threadDataPt[i] = node;
        }
    } else {
        threadDataPt = (pthreadBatchNode **)malloc(1*sizeof(pthreadBatchNode *));
        pthreadBatchNode *node = allocatePthreadBatchNode();
        node->max = max_array(ntLayers, numberOfLayers);
        node->weightsList = weightsList;
        node->biasesList = biasesList;
        node->activationsList = activationsList;
        node->zsList = zsList;;
        node->dcdwsList = initDcdwList(ntLayers, numberOfLayers);
        node->dcdbsList = initDcdbList(ntLayers, numberOfLayers);
        node->inoutSizes = inoutSizes;
        threadDataPt[0] = node;
    }
    
    // Stochastic gradient descent
    float **miniBatch = floatmatrix(0, miniBatchSize-1, 0, tr2-1);
    int delta;
    for (int k=1; k<=epochs; k++) {
        delta = 0;
        shuffle(trainingData, tr1, tr2);
        
        fprintf(stdout, "FeedforwardNT: Epoch {%d/%d}:\n", k, epochs);
        double rt = realtime();
        for (int l=1; l<=tr1/miniBatchSize; l++) {
            memcpy(*miniBatch, *trainingData+delta, (miniBatchSize*tr2)*sizeof(float));
            if (pthread) {
                updateMiniBatch(miniBatch, miniBatchSize, inoutSizes, weightsList, biasesList, dcdwsList, dcdbsList, threadDataPt, threadTID, ntLayers, numberOfLayers, tr1, eta, lambda, &pthread);
            } else {
                updateMiniBatch(miniBatch, miniBatchSize, inoutSizes, weightsList, biasesList, dcdwsList, dcdbsList, threadDataPt, NULL, ntLayers, numberOfLayers, tr1, eta, lambda, NULL);
            }
            delta = delta + ((int)miniBatchSize*(int)tr2);
        }
        rt = realtime() -  rt;
        fprintf(stdout, "FeedforwardNT: time to complete all training data set (s): %f\n", rt);
        
        if (testData != NULL) {
            fprintf(stdout, "FeedforwardNT: Epoch {%d/%d}: testing network with {%zu} inputs:\n", k, epochs, ts1);
            int result = evaluate(testData, ts1, inoutSizes, weightsList, biasesList, activationsList, zsList);
            fprintf(stdout, "FeedforwardNT: Epoch {%d/%d}: {%d} / {%zu}.\n", k, epochs, result, ts1);
        }
        
        float cost = totalCost(trainingData, tr1, weightsList, biasesList, activationsList, zsList, inoutSizes, classifications, lambda, false);
        fprintf(stdout, "FeedforwardNT: cost on training data: {%f}\n", cost);
        
        cost = totalCost(testData, ts1, weightsList, biasesList, activationsList, zsList, inoutSizes, classifications, lambda, true);
        fprintf(stdout, "FeedforwardNT: cost on test data: {%f}\n", cost);
        fprintf(stdout, "\n");
    }
    
    // Free-up memory
    
    deallocate(weightsList, biasesList, dcdwsList, dcdbsList, activationsList, zsList, threadDataPt, threadTID, miniBatchSize, pthread);
    free_fmatrix(miniBatch, 0, miniBatchSize-1, 0, tr2-1);
}
