#ifndef INIT_PLATFORM_H
#define INIT_PLATFORM_H

#include <CL/cl.h>
#include <stdio.h>

void init_OpenCl(cl_context *context, cl_command_queue *command_queue, int deviceSelected, cl_device_id *deviceID);

void exitOnFail(cl_int status, const char* message);

#endif //init_platform.h
