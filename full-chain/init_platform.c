#include "init_platform.h"


void init_OpenCl(cl_context *context, cl_command_queue *command_queue, int deviceSelected, cl_device_id *deviceID){

	cl_uint num_devs_returned;
	cl_context_properties properties[3];
	//cl_context context;
	//cl_command_queue command_queue;
	cl_uint numPlatforms;
    	cl_int status;
	cl_uint numDevices;
	cl_platform_id platformID;
    	//cl_device_id deviceID;
	int i, j, ok = 0;
	int isCPU = 0, isGPU = 0, isACCEL = 0;


    	status = clGetPlatformIDs(0, NULL, &numPlatforms); //num_platforms returns the number of OpenCL platforms available
    	exitOnFail(status, "number of platforms");

	cl_platform_id platformIDs[numPlatforms];
    	status = clGetPlatformIDs(numPlatforms, platformIDs, NULL); //platformsIDs returns a list of OpenCL platforms found. 
    	exitOnFail(status, "get platform IDs");


	if(deviceSelected == 0){
		isCPU=1;
	}
	else if(deviceSelected == 1){
		isGPU=1;
	}
	else if(deviceSelected == 2){
		isACCEL=1;
	}
	else{	
		printf("Selected device not found. Program will terminate\n");
		exit(1);
	}


	// iterate over platforms
	for (i = 0; i < numPlatforms; i++){
		// determine number of devices for a platform
		status = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		exitOnFail(status, "number of devices");
		if (CL_SUCCESS == status){
			// get device IDs for a platform
			//printf("Number of devices: %d\n", numDevices);
			cl_device_id deviceIDs[numDevices];
			status = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_ALL, numDevices, deviceIDs, NULL);
			if (CL_SUCCESS == status){
		       		// iterate over devices
		    		for (j = 0; j < numDevices && !ok; j++){
		       			cl_device_type deviceType;
			  		status = clGetDeviceInfo(deviceIDs[j], CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, NULL);
	            			if (CL_SUCCESS == status){
						//printf("Device Type: %d\n", deviceType);
						//CPU device
	               				if (isCPU && (CL_DEVICE_TYPE_CPU & deviceType)){
							ok=1;
	               					platformID = platformIDs[i];
	              					*deviceID = deviceIDs[j];
	               				}
				       		//GPU device
				       		if (isGPU && (CL_DEVICE_TYPE_GPU & deviceType)){
							ok=1;
							platformID = platformIDs[i];
							*deviceID = deviceIDs[j];
	                			}
						//ACCELERATOR device
	               				if (isACCEL && (CL_DEVICE_TYPE_ACCELERATOR & deviceType)){
							ok=1;
							platformID = platformIDs[i];
							*deviceID = deviceIDs[j];
	                			}
					}
	        		}
		    	}
		}
	} 

	if(!ok){
		printf("Selected device not found. Program will terminate\n");
		exit(-1);
	}

	*context = clCreateContext(NULL, 1, deviceID, NULL, NULL, &status);//Context
	exitOnFail( status, "clCreateContext" );

	*command_queue = clCreateCommandQueue(*context, *deviceID, CL_QUEUE_PROFILING_ENABLE, &status);
    	exitOnFail(status, "Error: Failed to create a command queue!");	

}

void exitOnFail(cl_int status, const char* message){
	if (CL_SUCCESS != status){
		printf("error: %s\n", message);
		printf("error: %d\n", status);
		exit(-1);
	}
}
