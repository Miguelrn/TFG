#include "sga.h"

#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable

//void exitOnFail(cl_int status, const char* message);


pos *sga_gpu(   double *imagen,
		int num_endmembers,
		int muestras, 
		int lineas, 
		int bandas,  
		double *endmember_bandas, 
		size_t localSize,
		cl_context context,
		cl_command_queue command_queue){


	pos *solucion = (pos*) malloc((num_endmembers)*sizeof(pos));
	int i,j;
	int *solu = (int*) calloc(num_endmembers * 2, sizeof(int));
	cl_mem ImageIn;
	cl_mem posiciones;
	cl_mem volumen;
	cl_mem ImageOut;
	int num_loop = 1;
	int primeraVuelta = 1;
	int ok=0;
	int widtheight = muestras*lineas;
	int matrixSize = (num_endmembers+1)*(num_endmembers+1);


	cl_program program;
	cl_kernel kernel_endmember;
	cl_kernel kernel_reduce;
	cl_kernel kernel_extrae;
    	cl_int status;
	cl_ulong start=(cl_ulong)0;
	cl_ulong end=(cl_ulong)0;
	cl_event ev_endmember, ev_reduce, ev_extrae, ev_memcpy;
	size_t global;
	global = ceil(muestras*lineas/(int)localSize)*localSize;//muestras*lineas;
	size_t globalSize_reduction = localSize;
	size_t globalSize_extraccion = bandas;
	double *volumenCPU = (double*) calloc (muestras * lineas, sizeof(double));

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

	kernel_src = (char*) malloc(sizeof(char)*(filelen+1));
	readlen = fread(kernel_src,1,filelen,fp);
	if(readlen!= filelen)
	{
		printf("error reading file\n");
		exit(1);
	}
	
	// ensure the string is NULL terminated
	kernel_src[filelen]='\0';
   	
	
	t0d = get_time();	
	 
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
	ImageIn  = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(double) * muestras * lineas * bandas, imagen, &status);
	exitOnFail(status, "create buffer d_image");
	posiciones = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  sizeof(int) * num_endmembers * 2, solu, &status);
	exitOnFail(status, "create buffer posiciones");
	t_device = get_time();
	//printf("\nTotal RAM->DEVICE:	\t%.5f (seconds)\n", t_device-t_ram);		

	volumen = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  sizeof(double) * muestras * lineas, volumenCPU, &status);
	exitOnFail(status, "create buffer volumen");
	ImageOut = clCreateBuffer(context,  CL_MEM_WRITE_ONLY,  sizeof(double) * bandas * num_endmembers, NULL, &status);
	exitOnFail(status, "create buffer ImageOut");
  

	status = clSetKernelArg(kernel_endmember, 0, sizeof(cl_mem), &ImageIn);
	status |= clSetKernelArg(kernel_endmember, 1, sizeof(cl_mem), &posiciones);
	status |= clSetKernelArg(kernel_endmember, 2, sizeof(cl_mem), &volumen);
	status |= clSetKernelArg(kernel_endmember, 3, sizeof(cl_uint), &num_loop);
	status |= clSetKernelArg(kernel_endmember, 4, sizeof(cl_uint), &muestras);
	status |= clSetKernelArg(kernel_endmember, 5, sizeof(cl_uint), &lineas);
	status |= clSetKernelArg(kernel_endmember, 6, sizeof(cl_double)*matrixSize, NULL);
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
 	status |= clSetKernelArg(kernel_extrae, 8, sizeof(cl_uint), &num_endmembers);
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

	status = clEnqueueReadBuffer(command_queue, ImageOut, CL_TRUE, 0, sizeof(double) * num_endmembers * bandas, endmember_bandas, 0, NULL, &ev_memcpy);
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
	printf("\nTotal INIT:	\t%.3f (seconds)\n", (t1d-t0d)-(tRamDevice));	
	tTotal = tRamDevice + k_endmember + k_reduce + k_extrae + read + write;
	printf("\nTotal SGA:	\t%.3f (seconds)\n RAM->Device: \t%.3f\t(%.1f%) \n endmember: \t%.3f\t(%.1f%) \n reduce: \t%.3f\t(%.1f%)\n", tTotal, tRamDevice, (tRamDevice*100)/tTotal, k_endmember, (k_endmember*100)/tTotal, k_reduce, (k_reduce*100)/tTotal);
	printf(" extract: \t%.3f\t(%.1f%)\n write: \t%.3f\t(%.1f%)\n read: \t\t%.3f\t(%.1f%)\n", k_extrae, (k_extrae*100)/tTotal, write, (write*100)/tTotal, read, (read/100)/tTotal);
	printf("\nTotal TIME:	\t%.3f (seconds)\n\n", (t1d-t0d) -(tRamDevice) + tTotal);
	
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
	clReleaseKernel(kernel_extrae);

	free(kernel_src);
	free(solu);

	return solucion;
}











