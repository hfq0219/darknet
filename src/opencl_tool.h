#ifndef OPENCL_TOOL_H
#define OPENCL_TOOL_H

#include "CL/cl.h"
#include <stdio.h>
#include <stdlib.h>

extern cl_platform_id *clPlatform;
extern cl_device_id *clDevice;
extern cl_context *clContext;
extern cl_command_queue *clCommandQueue;
extern cl_program *clProgram;
extern cl_kernel *clKernel;

int CreateTool(cl_platform_id *platform,cl_device_id *device,cl_context *context,
                cl_command_queue *commandQueue,cl_program *program,const char *fileName);

void clean(cl_context *context,cl_command_queue *commandQueue,cl_program *program,cl_kernel *kernel);

void setWorkItemSize(int kernel_num,size_t global_work_size[3],size_t local_work_size[3]);

#endif