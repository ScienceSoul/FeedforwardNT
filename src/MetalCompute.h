//
//  MetalCompute.h
//  FeedforwardNT
//
//  Created by Hakime Seddik on 08/06/2018.
//  Copyright © 2018 ScienceSoul. All rights reserved.
//

#ifndef MetalCompute_h
#define MetalCompute_h


typedef struct MetalCompute {
    
    void (* _Nullable init) (void);
} MetalCompute;

MetalCompute * _Nonnull metalCompute(void);

#endif /* MetalCompute_h */
