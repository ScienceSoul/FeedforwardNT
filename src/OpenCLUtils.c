//
//  OpenCLUtils.c
//  FeedforwardNT
//
//  Created by Seddik hakime on 10/06/2017.
//

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "OpenCLUtils.h"
#include "Memory.h"
#include "Utils.h"

cl_platform_id _Nullable * _Nullable find_platforms(cl_uint * _Nullable numberofPlatforms, cl_uint * _Nullable numberOfDevices) {
    
    cl_platform_id    *platForms;
    cl_int            err;
    cl_uint           num_platforms=0, num_devices=0, tot_devices=0;
    size_t            ext_size;
    char              *ext_data;
    
    // Get the platforms
    err = clGetPlatformIDs(1, NULL, &num_platforms);
    if (err < 0) {
        fprintf(stderr, "find_platforms: could not find any plateform.\n");
        fatal("find_platforms");
    }
    
    platForms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platForms, NULL);
    
    fprintf(stdout, "find_platforms: %d platform(s) available in system.\n", num_platforms);
    
    for (int i=0; i<num_platforms; i++) {
        
        err = clGetPlatformInfo(platForms[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &ext_size);
        if (err < 0) {
            fprintf(stderr, "find_platforms: could not read extension data.\n");
            fatal("find_platforms");
        }
        
        ext_data = (char *)malloc(ext_size);
        clGetPlatformInfo(platForms[i], CL_PLATFORM_EXTENSIONS, ext_size, ext_data, NULL);
        fprintf(stdout, "find_platforms: platform id:%d supports extensions: %s\n", i, ext_data);
        free(ext_data);
        
        err = clGetDeviceIDs(platForms[i], CL_DEVICE_TYPE_GPU, 1, NULL, &num_devices);
        if (err < 0) {
            fprintf(stderr, "find_platforms: could not find any device.\n");
            fatal("find_platforms");
        }
        fprintf(stdout, "find_platforms: %d built-in GPUs in platform id:%d.\n", num_devices, i);
        tot_devices = tot_devices + num_devices;
    }
    
    return platForms;
}

cl_device_id _Nullable find_single_device(void) {
    
    cl_platform_id    *platForms;
    cl_device_id      device;
    cl_uint           num_platforms=0, num_devices=0;
    
    // Get the platforms and the total number of devices
    // in the system
    platForms = find_platforms(&num_platforms, &num_devices);
    if (num_devices > 1) fprintf(stdout, "OpenCL: several devices found but only use one of them.\n");
    clGetDeviceIDs(platForms[0], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    free(platForms);
    
    return device;
}

int device_info(cl_device_id _Nonnull device_id) {
    
    cl_uint err, clock_frequency, address_bits;
    
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1025] = {0};
    cl_char device_profile[1024] = {0};
    cl_char device_driver_version[1024] = {0};
    
    err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    
    err |= clGetDeviceInfo(device_id, CL_DRIVER_VERSION, sizeof(device_driver_version), device_driver_version, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_PROFILE, sizeof(device_profile), device_profile, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_ADDRESS_BITS, sizeof(address_bits), &address_bits, NULL);
    
    fprintf(stdout,"device_info: vendor: %s.\n", vendor_name);
    fprintf(stdout,"device_info: device name: %s.\n", device_name);
    fprintf(stdout,"device_info: OpenCL (driver) version: %s.\n", device_driver_version);
    fprintf(stdout,"device_info: supported profile: %s.\n", device_profile);
    fprintf(stdout,"device_info: clock frequency (MHz): %u.\n", clock_frequency);
    fprintf(stdout,"device_info: address space size (bits): %u.\n", address_bits);
    
    return CL_SUCCESS;
}

int device_stats(cl_device_id _Nonnull device_id) {
    
    cl_uint err, addr_size, clock_frequency, max_compute_units;
    
    cl_char device_extensions[1024] = {0};
    cl_uint vector_width_native, vector_width_prefered;
    
    cl_ulong global_mem_size, global_mem_cache_size, local_mem_size;
    cl_ulong max_mem_alloc_size;
    
    size_t max_work_group_size, max_work_item_sizes[3];
    
    cl_uint vector_width_types_native[] = {CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE};
    
    cl_uint vector_width_types_prefered[] = {CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE};
    
    char *vector_type_names[] = {"char", "short", "int", "long", "float", "double"};
    
    err = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, sizeof(device_extensions), device_extensions, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_ADDRESS_BITS, sizeof(addr_size), &addr_size, NULL);
    
    err |= clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(global_mem_cache_size), &global_mem_cache_size, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
    
    err |= clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units), &max_compute_units, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), max_work_item_sizes, NULL);
    
    fprintf(stdout,"device_stats: supported OpenCL extensions: %s.\n",device_extensions);
    fprintf(stdout,"device_stats: address width: %u.\n",addr_size);
    fprintf(stdout,"device_stats: global memory size (MB): %llu.\n", global_mem_size/(1024*1024));
    fprintf(stdout,"device_stats: local memory size (KB): %llu.\n",local_mem_size/1024);
    fprintf(stdout,"device_stats: global memory cache line size (KB): %llu.\n", global_mem_cache_size/1024);
    fprintf(stdout,"device_stats: maximum memory allocation size (MB): %llu.\n", max_mem_alloc_size/(1024*1024));
    fprintf(stdout,"device_stats: maximum compute units: %u.\n", max_compute_units);
    fprintf(stdout,"device_stats: clock frequency (MHz): %u.\n", clock_frequency);
    
    fprintf(stdout,"device_stats: maximum work group size: %zu.\n", max_work_group_size);
    fprintf(stdout,"device_stats: maximum work item size (each dimension): %zu %zu %zu.\n", max_work_item_sizes[0], max_work_item_sizes[1], max_work_item_sizes[2]);
    
    for(int k=0;k<6;k++) {
        err |= clGetDeviceInfo(device_id, vector_width_types_native[k], sizeof(vector_width_native), &vector_width_native, NULL);
        fprintf(stdout,"device_stats: native vector type width for: %s = %u.\n", vector_type_names[k], vector_width_native);
    }
    for(int k=0;k<6;k++) {
        err |= clGetDeviceInfo(device_id, vector_width_types_prefered[k], sizeof(vector_width_prefered), &vector_width_prefered, NULL);
        fprintf(stdout,"device_stats: prefered vector type width for: %s = %u.\n", vector_type_names[k], vector_width_prefered);
    }
    
    return CL_SUCCESS;
}
