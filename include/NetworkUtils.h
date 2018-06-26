//
//  NetworkUtils.h
//  FeedforwardNT
//
//  Created by Hakime Seddik on 26/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#ifndef NetworkUtils_h
#define NetworkUtils_h

void * _Nonnull allocateActivationNode(void);
void * _Nonnull allocateZNode(void);
void * _Nonnull allocateDcdwNode(void);
void * _Nonnull allocateDcdbNode(void);

float * _Nonnull initWeights(int * _Nonnull ntLayers, unsigned int numberOfLayers);
float * _Nonnull initBiases(int * _Nonnull ntLayers, unsigned int numberOfLayers);

void * _Nonnull initActivationsList(int * _Nonnull ntLayers, unsigned int numberOfLayers);
void * _Nonnull initZsList(int * _Nonnull ntLayers, unsigned int numberOfLayers);

void * _Nonnull initDcdwList(int * _Nonnull ntLayers, unsigned int numberOfLayers);
void * _Nonnull initDcdbList(int * _Nonnull ntLayers, unsigned int numberOfLayers);

int loadParameters(void * _Nonnull self, const char * _Nonnull paraFile);

#endif /* NetworkUtils_h */
