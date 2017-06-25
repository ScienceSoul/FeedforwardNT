//
//  OpenCLUtils.h
//  FeedforwardNT
//
//  Created by Seddik hakime on 10/06/2017.
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

cl_platform_id _Nullable * _Nullable find_platforms(cl_uint * _Nullable numberofPlatforms, cl_uint * _Nullable numberOfDevices);
cl_device_id _Nullable find_single_device(void);
int device_info(cl_device_id _Nonnull device_id);
int device_stats(cl_device_id _Nonnull device_id);
int LoadFileIntoString(const char * _Nonnull filename, char * _Nonnull * _Nullable text, size_t * _Nonnull len);

#endif
