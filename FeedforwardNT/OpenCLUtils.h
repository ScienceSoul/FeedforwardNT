//
//  OpenCLUtils.h
//  FeedforwardNT
//
//  Created by Seddik hakime on 10/06/2017.
//  Copyright © 2017 ScienceSoul. All rights reserved.
//

#ifndef OpenCLUtils_h
#define OpenCLUtils_h

#ifdef USE_OPENCL_GPU
#ifdef __APPLE__
    #include <OpenCL/OpenCL.h>
#else
    #include <CL/cl.h>
#endif
#endif

#endif /* OpenCLUtils_h */

#ifdef USE_OPENCL_GPU
cl_platform_id __nullable * __nullable find_platforms(cl_uint * __nullable numberofPlatforms, cl_uint * __nullable numberOfDevices);
cl_device_id __nullable find_single_device(void);
int device_info(cl_device_id __nonnull device_id);
int device_stats(cl_device_id __nonnull device_id);
int LoadFileIntoString(const char * __nonnull filename, char * __nonnull * __nullable text, size_t * __nonnull len);
#endif
