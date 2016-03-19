#include "sga.h"

#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable




pos *sga_gpu(float *imagen, int num_endmembers, int muestras, int lineas, int bandas, int deviceSelected, float *endmember_bandas, double treadImage, size_t localSize){

	pos *solucion = (pos*) malloc((num_endmembers)*sizeof(pos));
	int i,j;
	int *solu = (int*) calloc(num_endmembers * 2, sizeof(int));
	cl_mem ImageIn;
	cl_mem posiciones;
	cl_mem volumen;
	cl_mem matrix_aux;
	cl_mem ImageOut;
	int num_loop = 1;
	int primeraVuelta = 1;
	int ok=0;
	int widtheight = muestras*lineas;
	int matrixSize = (num_endmembers+1)*(num_endmembers+1);

	cl_uint num_devs_returned;
	cl_context_properties properties[3];
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel_endmember;
	cl_kernel kernel_reduce;
	cl_kernel kernel_extrae;
    	cl_int status;
	cl_ulong start=(cl_ulong)0;
	cl_ulong end=(cl_ulong)0;
	cl_event ev_endmember, ev_reduce, ev_extrae, ev_memcpy;
	//size_t localSize = 64, local_size_size;//probar a traerlo por parametro??
	size_t global;
	global = ceil(muestras*lineas/(float)localSize)*localSize;//muestras*lineas;
	size_t globalSize_reduction = localSize;//muestras...si es muy pequeño igual interesa ponerlo en cpu la reduccion final
	size_t globalSize_extraccion = bandas;

	double t0d, t1d, t1fin, t_ram, t_device;
	double k_endmember = 0.0, k_reduce = 0.0, k_extrae = 0.0, read = 0.0, write = 0.0, tTotal = 0.0, tRamDevice=0.0;
	
	FILE *fp;
	long filelen;
	long readlen;
	char *kernel_src;  //char string to hold kernel source
	
	fp = fopen("Hyperespectral.cl","r");
	fseek(fp,0L, SEEK_END);
	filelen = ftell(fp);
	rewind(fp);

	kernel_src = malloc(sizeof(char)*(filelen+1));
	readlen = fread(kernel_src,1,filelen,fp);
	if(readlen!= filelen)
	{
		printf("error reading file\n");
		exit(1);
	}
	
	// ensure the string is NULL terminated
	kernel_src[filelen+1]='\0';

    	// determine number of platforms
    	cl_uint numPlatforms;
    	status = clGetPlatformIDs(0, NULL, &numPlatforms); //num_platforms returns the number of OpenCL platforms available
    	exitOnFail(status, "number of platforms");
	if (CL_SUCCESS == status){
		printf("\nNumber of OpenCL platforms: %d\n", numPlatforms);
		printf("\n-------------------------\n");
	}
	
	// get platform IDs
  	cl_platform_id platformIDs[numPlatforms];
    	status = clGetPlatformIDs(numPlatforms, platformIDs, NULL); //platformsIDs returns a list of OpenCL platforms found. 
    	exitOnFail(status, "get platform IDs");

	cl_uint numDevices;
	cl_platform_id platformID;
    	cl_device_id deviceID;
	
	//deviceSelected-> 0:CPU, 1:GPU, 2:ACCELERATOR
	int isCPU = 0, isGPU = 0, isACCEL=0;
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
			printf("Number of devices: %d\n", numDevices);
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
	              					deviceID = deviceIDs[j];
	               				}
				       		//GPU device
				       		if (isGPU && (CL_DEVICE_TYPE_GPU & deviceType)){
							ok=1;
							platformID = platformIDs[i];
							deviceID = deviceIDs[j];
	                			}
						//ACCELERATOR device
	               				if (isACCEL && (CL_DEVICE_TYPE_ACCELERATOR & deviceType)){
							ok=1;
							platformID = platformIDs[i];
							deviceID = deviceIDs[j];
	                			}
					}
	        		}
		    	}
		}
	} 

	if(!ok){
		printf("Selected device not found. Program will terminate\n");
		return 1;
	}	
	
	t0d = get_time();	

	context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &status);//Context
	exitOnFail( status, "clCreateContext" );
	
	// Create a command queue
	command_queue = clCreateCommandQueue(context, deviceID, CL_QUEUE_PROFILING_ENABLE, &status);
    	exitOnFail(status, "Error: Failed to create a command queue!");
	 
	// create program object from source. 
	// kernel_src contains source read from file earlier
	program = clCreateProgramWithSource(context, 1 ,(const char **) & kernel_src, NULL, &status);
	exitOnFail(status, "Unable to create program object.");       

	status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	exitOnFail(status, "Build failed.");	

	kernel_endmember = clCreateKernel(program, "endmembers_calculation", &status);
	exitOnFail(status, "Unable to create kernel calculation object.");

	kernel_reduce = clCreateKernel(program, "reduce", &status);
	exitOnFail(status, "Unable to create kernel reduce object.");

	kernel_extrae = clCreateKernel(program, "extrae_endmember", &status);
	exitOnFail(status, "Unable to create kernel extrae_endmember object.");

	srand(time(NULL));
	solu[1] = rand() % lineas;//221
 	solu[0] = rand() % muestras;//325

	t_ram = get_time();
	ImageIn  = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(float) * muestras * lineas * bandas, imagen, &status);
	exitOnFail(status, "create buffer d_image");
	posiciones = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  sizeof(int) * num_endmembers * 2, solu, &status);
	exitOnFail(status, "create buffer posiciones");
	t_device = get_time();
	//printf("\nTotal RAM->DEVICE:	\t%.5f (seconds)\n", t_device-t_ram);		

	volumen = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(double) * muestras * lineas, NULL, &status);
	exitOnFail(status, "create buffer volumen");
	matrix_aux = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(double) * muestras * lineas, NULL, &status);
	exitOnFail(status, "create buffer matrix_aux");
	ImageOut = clCreateBuffer(context,  CL_MEM_WRITE_ONLY,  sizeof(float) * bandas * num_endmembers, NULL, &status);
	exitOnFail(status, "create buffer matrix_aux");
  

	status = clSetKernelArg(kernel_endmember, 0, sizeof(cl_mem), &ImageIn);
	status |= clSetKernelArg(kernel_endmember, 1, sizeof(cl_mem), &posiciones);
	status |= clSetKernelArg(kernel_endmember, 2, sizeof(cl_mem), &volumen);
	status |= clSetKernelArg(kernel_endmember, 3, sizeof(cl_uint), &num_loop);
	status |= clSetKernelArg(kernel_endmember, 4, sizeof(cl_uint), &muestras);
	status |= clSetKernelArg(kernel_endmember, 5, sizeof(cl_uint), &lineas);
	status |= clSetKernelArg(kernel_endmember, 6, sizeof(cl_double)*matrixSize, NULL);
	status |= clSetKernelArg(kernel_endmember, 7, sizeof(cl_double), &matrix_aux);
	exitOnFail(status, "Unable to set kernel calculation arguments.");

	status = clSetKernelArg(kernel_reduce, 0, sizeof(cl_mem), &posiciones);
    	status |= clSetKernelArg(kernel_reduce, 1, sizeof(cl_mem), &volumen);
 	status |= clSetKernelArg(kernel_reduce, 2, sizeof(cl_uint), &num_loop); 
 	status |= clSetKernelArg(kernel_reduce, 3, sizeof(cl_uint), &primeraVuelta);
    	status |= clSetKernelArg(kernel_reduce, 4, sizeof(cl_double)*localSize, NULL);
    	status |= clSetKernelArg(kernel_reduce, 5, sizeof(cl_int)*localSize, NULL);
 	status |= clSetKernelArg(kernel_reduce, 6, sizeof(cl_uint), &muestras);
	status |= clSetKernelArg(kernel_reduce, 7, sizeof(cl_uint), &lineas);
	exitOnFail(status, "Unable to set kernel reduce arguments.");

	status = clSetKernelArg(kernel_extrae, 0, sizeof(cl_mem), &posiciones);
    	status |= clSetKernelArg(kernel_extrae, 1, sizeof(cl_mem), &ImageIn);
    	status |= clSetKernelArg(kernel_extrae, 2, sizeof(cl_mem), &ImageOut);
 	status |= clSetKernelArg(kernel_extrae, 3, sizeof(cl_uint), &num_loop); 
 	status |= clSetKernelArg(kernel_extrae, 4, sizeof(cl_uint), &primeraVuelta);
 	status |= clSetKernelArg(kernel_extrae, 5, sizeof(cl_uint), &muestras); 
 	status |= clSetKernelArg(kernel_extrae, 6, sizeof(cl_uint), &lineas);
 	status |= clSetKernelArg(kernel_extrae, 7, sizeof(cl_uint), &bandas); 
	exitOnFail(status, "Unable to set kernel extrae arguments.");

	//printf("Global: %d, Local: %d\n",global,localSize);
	t1d = get_time();
	
	while(num_loop < num_endmembers){
		status = clEnqueueNDRangeKernel(command_queue, kernel_endmember, 1, NULL, &global, &localSize, 0, NULL, &ev_endmember);
		exitOnFail(status, "Launch OpenCL endmember kernel");
		start=(cl_ulong)0;
		end=(cl_ulong)0;
		clWaitForEvents(1,&ev_endmember);
		status = clGetEventProfilingInfo(ev_endmember, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
		exitOnFail(status, "Profiling kernel endmember - start");
		status=clGetEventProfilingInfo(ev_endmember, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
		exitOnFail(status, "Profiling kernel endmember - end");
		clReleaseEvent(ev_endmember);
		k_endmember+=(end-start)*1.0e-9;

		status = clEnqueueNDRangeKernel(command_queue, kernel_reduce, 1, NULL, &globalSize_reduction, &localSize, 0, NULL, &ev_reduce);
		exitOnFail(status, "Launch OpenCL reduction kernel");
		start=(cl_ulong)0;
		end=(cl_ulong)0;
		clWaitForEvents(1,&ev_reduce);
		status = clGetEventProfilingInfo(ev_reduce, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
		exitOnFail(status, "Profiling kernel reduction - start");
		status=clGetEventProfilingInfo(ev_reduce, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
		exitOnFail(status, "Profiling kernel reduction - end");
		clReleaseEvent(ev_reduce);
		k_reduce+=(end-start)*1.0e-9;

		status = clEnqueueNDRangeKernel(command_queue, kernel_extrae, 1, NULL, &globalSize_extraccion, NULL, 0, NULL, &ev_extrae);
		exitOnFail(status, "Launch OpenCL extraccion kernel");
		start=(cl_ulong)0;
		end=(cl_ulong)0;
		clWaitForEvents(1,&ev_extrae);
		status = clGetEventProfilingInfo(ev_extrae, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
		exitOnFail(status, "Profiling kernel extraccion - start");
		status=clGetEventProfilingInfo(ev_extrae, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
		exitOnFail(status, "Profiling kernel extraccion - end");
		clReleaseEvent(ev_extrae);
		k_extrae+=(end-start)*1.0e-9;


		if(primeraVuelta){
			primeraVuelta--;
            		clSetKernelArg(kernel_reduce, 3, sizeof(cl_uint), &primeraVuelta);
			clSetKernelArg(kernel_extrae, 4, sizeof(cl_uint), &primeraVuelta);

		}
		else{
			num_loop++;
        		clSetKernelArg(kernel_endmember, 3, sizeof(cl_uint), &num_loop);
       			clSetKernelArg(kernel_reduce, 2, sizeof(cl_uint), &num_loop);
			clSetKernelArg(kernel_extrae, 3, sizeof(cl_uint), &num_loop);
		}
	}
	t1fin = get_time();

	// wait for the command to finish
	clFinish(command_queue);

	// read the output back to host memory
	status = clEnqueueReadBuffer(command_queue, posiciones, CL_TRUE, 0, sizeof(int) * num_endmembers * 2, solu, 0, NULL, &ev_memcpy);
	exitOnFail(status, "Error enqueuing read buffer command.");
	start=(cl_ulong)0;
	end=(cl_ulong)0;
	clWaitForEvents(1,&ev_memcpy);
	status = clGetEventProfilingInfo(ev_memcpy, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	exitOnFail(status, "Profiling kernel reduction - start");
	status=clGetEventProfilingInfo(ev_memcpy, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	exitOnFail(status, "Profiling kernel reduction - end");
	clReleaseEvent(ev_memcpy);
	read+=(end-start)*1.0e-9;

	status = clEnqueueReadBuffer(command_queue, ImageOut, CL_TRUE, 0, sizeof(float) * num_endmembers * bandas, endmember_bandas, 0, NULL, &ev_memcpy);
	exitOnFail(status, "Error enqueuing read buffer command.");
	start=(cl_ulong)0;
	end=(cl_ulong)0;
	clWaitForEvents(1,&ev_memcpy);
	status = clGetEventProfilingInfo(ev_memcpy, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	exitOnFail(status, "Profiling kernel reduction - start");
	status=clGetEventProfilingInfo(ev_memcpy, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	exitOnFail(status, "Profiling kernel reduction - end");
	clReleaseEvent(ev_memcpy);
	read+=(end-start)*1.0e-9;

	tRamDevice = (t_device-t_ram);
	printf("\nTotal INIT:	\t%.3f (seconds)\n", (t1d-t0d)+treadImage-(tRamDevice));	
	tTotal = tRamDevice + k_endmember + k_reduce + k_extrae + read + write;
	printf("\nTotal SGA:	\t%.3f (seconds)\n RAM->Device: \t%.3f\t(%.1f%) \n endmember: \t%.3f\t(%.1f%) \n reduce: \t%.3f\t(%.1f%)\n", tTotal, tRamDevice, (tRamDevice*100)/tTotal, k_endmember, (k_endmember*100)/tTotal, k_reduce, (k_reduce*100)/tTotal);
	printf(" extract: \t%.3f\t(%.1f%)\n write: \t%.3f\t(%.1f%)\n read: \t\t%.3f\t(%.1f%)\n", k_extrae, (k_extrae*100)/tTotal, write, (write*100)/tTotal, read, (read/100)/tTotal);
	printf("\nTotal TIME:	\t%.3f (seconds)\n\n", (t1d-t0d)+ treadImage -(tRamDevice) + tTotal);
	
	for(i = 0;i < num_endmembers; i++){
		solucion[i].filas = solu[i*2]; 
		solucion[i].columnas = solu[i*2+1];
	}

    	clReleaseMemObject(ImageIn);
    	clReleaseMemObject(posiciones);
    	clReleaseMemObject(volumen);

	clReleaseProgram(program);
	clReleaseKernel(kernel_endmember);
	clReleaseKernel(kernel_reduce);
	clReleaseCommandQueue(command_queue);
    	clReleaseContext(context);

	free(kernel_src);
	free(solu);//comprobar si falta algun free mas !1

	return solucion;
}

double get_time(){
	static struct timeval tv0;
	double time_, time;

	gettimeofday(&tv0,(struct timezone*)0);
	time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
	time = time_/1000000;
	return(time);
}

void exitOnFail(cl_int status, const char* message){
	if (CL_SUCCESS != status){
		printf("error: %s\n", message);
		printf("error: %d\n", status);
		exit(-1);
	}
}







