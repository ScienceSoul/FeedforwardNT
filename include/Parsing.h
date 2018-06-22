//
//  Parsing.h
//  FeedforwardNT
//
//  Created by Hakime Seddik on 22/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#ifndef Parsing_h
#define Parsing_h

#include <stdbool.h>

#define MAX_KEY_VALUE_STRING 1024

typedef struct dictionary {
    
    bool has_tag;
    char key[MAX_KEY_VALUE_STRING];
    char value[MAX_KEY_VALUE_STRING];
    char tag[1];
    struct dictionary * _Nullable next;
    struct dictionary * _Nullable previous;
    
} dictionary;

typedef struct definition {
    
    int def_id;
    int number_of_fields;
    dictionary * _Nullable field;
    struct definition * _Nullable next;
    struct definition *_Nullable previous;
    
} definition;

definition * _Nonnull allocateDefinitionNode(void);
dictionary * _Nonnull allocateDictionaryNode(void);

definition * _Nullable getDefinitions(void * _Nonnull neural, const char * _Nonnull paramsDefFile, const char * _Nonnull keyword);


#endif /* Parsing_h */
