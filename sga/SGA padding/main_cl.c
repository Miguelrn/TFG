#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <CL/cl.h>
#include <math.h>
#include <mm_malloc.h>


#define PADDING 4

#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable
#define DEBUG 0
#define MAXCAD 100

//scp TFG/OpenCL/main_cl.c mrnavarro@geforce.dacya.ucm.es:/home/mrnavarro/TFG/OpenCL/main_cl.c
typedef struct{
	int filas;
	int columnas;
}pos;


double get_time();
double *lectura_archivo(char *ruta, int *lineas, int *muestras, int *bandas, char *tipo);
pos *sga_gpu(float *imagenPAD, int num_endmembers, int muestras, int lineas, int bandas, int deviceSelected, float *endmember_bandas, double treadImage, size_t localSize, int widtheightPAD, int bandsPAD);
void exitOnFail(cl_int status, const char* message);

//-------------------------------------------Open Cl -------------------------------------------------//
pos *sga_gpu(float *imagenPAD, int num_endmembers, int muestras, int lineas, int bandas, int deviceSelected, float *endmember_bandas, double treadImage, size_t localSize, int widtheightPAD, int bandsPAD){

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

	//printf("widtheightPAD: %d, bandsPAD: %d\n", widtheightPAD, bandsPAD);

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
	size_t globalSize_reduction = localSize;
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
	ImageIn  = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  sizeof(float) * widtheightPAD * bandsPAD, imagenPAD, &status);
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
	status |= clSetKernelArg(kernel_endmember, 6, sizeof(cl_uint), &bandas);
	status |= clSetKernelArg(kernel_endmember, 7, sizeof(cl_double)*matrixSize, NULL);
	status |= clSetKernelArg(kernel_endmember, 8, sizeof(cl_double), &matrix_aux);
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

//----------------------------------------fin Open Cl -------------------------------------------------//

int main(int argc, char **argv) {


	//Variables para calcular el tiempo
	double t_0,t0, t1, treadImage;

	float *imagen;
	char *tipo = (char*)malloc(MAXCAD*sizeof(char));
	char *imagenhdr = (char*)malloc(MAXCAD*sizeof(char));
	char *imagenbsq = (char*)malloc(MAXCAD*sizeof(char));

	int lines, samples, bands, datatype, i,j;
	int endmember, error, deviceSelected;
	size_t localSize;
	pos *solucion;

	//Tener menos de 4 argumentos es incorrecto
	if (argc < 5) {
		fprintf(stderr, "Uso incorrecto de los parametros ./exe 'ruta imagen' 'numero de Endmemebers' 'local size' 'deviceSelected (0|1|2)'\n");
		exit(1);
	}

	endmember = atoi(argv[2]);
	localSize = atoi(argv[3]);
	deviceSelected = atoi(argv[4]);

	t_0 = get_time();
	strcpy(imagenhdr,argv[1]);
	strcat(imagenhdr, ".hdr");
	readHeader(imagenhdr, &samples, &lines, &bands, &datatype);
    	printf("Lines = %d  Samples = %d  Nbands = %d  Data_type = %d\n", lines, samples, bands, datatype);
	//imagen = (float*) malloc(samples*lines*bands*sizeof(float));
	int widtheight = samples*lines;
	int widtheightPAD = (widtheight) + ((PADDING - widtheight % PADDING) % PADDING);
	int bandsPAD = (bands) + ((PADDING - bands % PADDING) % PADDING);
	float *imagenPAD = (float*) _mm_malloc(widtheightPAD * bandsPAD * sizeof(float), 64);

	strcpy(imagenbsq,argv[1]);
	strcat(imagenbsq, ".bsq");
	Load_Image(imagenbsq, imagenPAD, widtheight, widtheightPAD, bands, bandsPAD, datatype);
	float *endmember_bandas = (float*) calloc(bands*endmember, sizeof(float));
	t0 = get_time();
	treadImage=t0-t_0;
	solucion = sga_gpu(imagenPAD, endmember, samples, lines, bands, deviceSelected, endmember_bandas, treadImage, localSize, widtheightPAD, bandsPAD);
	t1 = get_time();

	//printf("\nTotal read files time: %f (seconds)\n", t0-t_0);


	strcpy(imagenbsq,argv[1]);
	strcat(imagenbsq, "SGAResult.bsq");
	writeResult(endmember_bandas, imagenbsq, endmember, 1, bands);//falta cambiarlo
	printf("File with endmembers saved at: %s\n",imagenbsq);

	for(i = 0; i < endmember; i++){
		printf("%2d: %d - %d\n",i+1,solucion[i].filas, solucion[i].columnas);
	}

	free(imagenPAD);
	free(tipo);
	free(imagenhdr);
	free(imagenbsq);
	free(endmember_bandas);
	
	return 0;

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
	switch (status) {

		case CL_SUCCESS :
		    printf(" CL_SUCCESS ");break;
		case CL_DEVICE_NOT_FOUND :
		     printf(" CL_DEVICE_NOT_FOUND ");break;
		case CL_DEVICE_NOT_AVAILABLE :
		     printf(" CL_DEVICE_NOT_AVAILABLE ");break;
		case CL_COMPILER_NOT_AVAILABLE :
		     printf(" CL_COMPILER_NOT_AVAILABLE ");break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE :
		     printf(" CL_MEM_OBJECT_ALLOCATION_FAILURE ");break;
		case CL_OUT_OF_RESOURCES :
		     printf(" CL_OUT_OF_RESOURCES ");break;
		case CL_OUT_OF_HOST_MEMORY :
		    printf(" CL_OUT_OF_HOST_MEMORY ");break;
		case CL_PROFILING_INFO_NOT_AVAILABLE :
		    printf(" CL_PROFILING_INFO_NOT_AVAILABLE ");break;
		case CL_MEM_COPY_OVERLAP :
		    printf(" CL_MEM_COPY_OVERLAP ");break;
		case CL_IMAGE_FORMAT_MISMATCH :
		    printf(" CL_IMAGE_FORMAT_MISMATCH ");break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED :
		    printf(" CL_IMAGE_FORMAT_NOT_SUPPORTED ");break;
		case CL_BUILD_PROGRAM_FAILURE :
		    printf(" CL_BUILD_PROGRAM_FAILURE ");break;
		case CL_MAP_FAILURE :
		    printf(" CL_MAP_FAILURE ");break;
		case CL_MISALIGNED_SUB_BUFFER_OFFSET :
		    printf(" CL_MISALIGNED_SUB_BUFFER_OFFSET ");break;
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST :
		    printf(" CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ");break;
		case CL_INVALID_VALUE :
		    printf(" CL_INVALID_VALUE ");break;
		case CL_INVALID_DEVICE_TYPE :
		    printf(" CL_INVALID_DEVICE_TYPE ");break;
		case CL_INVALID_PLATFORM :
		    printf(" CL_INVALID_PLATFORM ");break;
		case CL_INVALID_DEVICE :
		    printf(" CL_INVALID_DEVICE ");break;
		case CL_INVALID_CONTEXT :
		    printf(" CL_INVALID_CONTEXT ");break;
		case CL_INVALID_QUEUE_PROPERTIES :
		    printf(" CL_INVALID_QUEUE_PROPERTIES ");break;
		case CL_INVALID_COMMAND_QUEUE :
		    printf(" CL_INVALID_COMMAND_QUEUE ");break;
		case CL_INVALID_HOST_PTR :
		    printf(" CL_INVALID_HOST_PTR ");break;
		case CL_INVALID_MEM_OBJECT :
		    printf(" CL_INVALID_MEM_OBJECT ");break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR :
		    printf(" CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ");break;
		case CL_INVALID_IMAGE_SIZE :
		    printf(" CL_INVALID_IMAGE_SIZE ");break;
		case CL_INVALID_SAMPLER :
		    printf(" CL_INVALID_SAMPLER ");break;
		case CL_INVALID_BINARY :
		    printf(" CL_INVALID_BINARY ");break;
		case CL_INVALID_BUILD_OPTIONS :
		    printf(" CL_INVALID_BUILD_OPTIONS ");break;
		case CL_INVALID_PROGRAM :
		    printf(" CL_INVALID_PROGRAM ");break;
		case CL_INVALID_PROGRAM_EXECUTABLE :
		    printf(" CL_INVALID_PROGRAM_EXECUTABLE ");break;
		case CL_INVALID_KERNEL_NAME :
		    printf(" CL_INVALID_KERNEL_NAME ");break;
		case CL_INVALID_KERNEL_DEFINITION :
		    printf(" CL_INVALID_KERNEL_DEFINITION ");break;
		case CL_INVALID_KERNEL :
		    printf(" CL_INVALID_KERNEL ");break;
		case CL_INVALID_ARG_INDEX :
		    printf(" CL_INVALID_ARG_INDEX ");break;
		case CL_INVALID_ARG_VALUE :
		    printf(" CL_INVALID_ARG_VALUE ");break;
		case CL_INVALID_ARG_SIZE :
		    printf(" CL_INVALID_ARG_SIZE ");break;
		case CL_INVALID_KERNEL_ARGS :
		    printf(" CL_INVALID_KERNEL_ARGS ");break;
		case CL_INVALID_WORK_DIMENSION :
		    printf(" CL_INVALID_WORK_DIMENSION ");break;
		case CL_INVALID_WORK_GROUP_SIZE :
		    printf(" CL_INVALID_WORK_GROUP_SIZE ");break;
		case CL_INVALID_WORK_ITEM_SIZE :
		    printf(" CL_INVALID_WORK_ITEM_SIZE ");break;
		case CL_INVALID_GLOBAL_OFFSET :
		    printf(" CL_INVALID_GLOBAL_OFFSET ");break;
		case CL_INVALID_EVENT_WAIT_LIST :
		    printf(" CL_INVALID_EVENT_WAIT_LIST ");break;
		case CL_INVALID_EVENT :
		    printf(" CL_INVALID_EVENT ");break;
		case CL_INVALID_OPERATION :
		    printf(" CL_INVALID_OPERATION ");break;
		case CL_INVALID_GL_OBJECT :
		    printf(" CL_INVALID_GL_OBJECT ");break;
		case CL_INVALID_BUFFER_SIZE :
		    printf(" CL_INVALID_BUFFER_SIZE ");break;
		case CL_INVALID_MIP_LEVEL :
		    printf(" CL_INVALID_MIP_LEVEL ");break;
		case CL_INVALID_GLOBAL_WORK_SIZE :
		    printf(" CL_INVALID_GLOBAL_WORK_SIZE ");break;
		case CL_INVALID_PROPERTY :
		    printf(" CL_INVALID_PROPERTY ");break;
		default:
		    printf("UNKNOWN ERROR");break;

	    }
		exit(-1);
	}
}



