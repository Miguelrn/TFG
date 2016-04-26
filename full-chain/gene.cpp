#include "gene.h"

//export PATH=/usr/local/cuda-7.5/bin:$PATH
//export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH



int est_noise(double *image, magmaDouble_ptr image_Device, int linessamples, int bands, magmaDouble_ptr noise_Device, magma_queue_t queue){

	int info;
	double alpha = 1, beta = 0;
	int lwork = bands*bands;
	int i = 0, j = 0, b = 0;
	int uno = 1;
	size_t size = 0;

	//Device
	magmaDouble_ptr rr_Device, beta_Device;
	MALLOC_DEVICE(rr_Device, double, bands*bands)
	MALLOC_DEVICE(beta_Device, double, bands*bands)
	
	//Host
	double *rr_Host, *rri_Host, *xx_Host, *rra_Host, *beta_Host, *work_Host;
	magma_int_t *ipiv_Host;
	MALLOC_HOST(rr_Host, double, bands*bands)
	MALLOC_HOST(rri_Host, double, bands*bands) 
	MALLOC_HOST(xx_Host, double, bands*bands) 
	MALLOC_HOST(rra_Host, double, bands) 
	MALLOC_HOST(beta_Host, double, bands*bands) 
	MALLOC_HOST(ipiv_Host, magma_int_t, bands) 
	MALLOC_HOST(work_Host, double, bands*bands) 



	magma_dgemm(MagmaTrans, MagmaNoTrans, bands, bands, linessamples, alpha, image_Device, size, linessamples, image_Device, size, linessamples, beta,  rr_Device, size, bands, queue);

	magma_dgetmatrix(bands, bands, rr_Device, size, bands, rr_Host, bands, queue);



	for (i = 0; i < bands; i++){
		rr_Host[i*bands + i] = rr_Host[i*bands + i]  + 1e-6;
	}

	memcpy(rri_Host, rr_Host, bands*bands*sizeof(double));

	lapackf77_dgetrf(&bands, &bands, rri_Host, &bands, ipiv_Host, &info);
	lapackf77_dgetri(&bands, rri_Host, &bands, ipiv_Host, work_Host, &lwork, &info);


	for (b = 0; b < bands; b++){//se podria mirar si interesa usar magma
		for (i = 0; i < bands; i++){
			for (j = 0; j < bands;j++){
				xx_Host[i*bands +j] = rri_Host[i*bands +j] - ( (rri_Host[i*bands+b] * rri_Host[b*bands+j]) / rri_Host[b*bands + b]);
			}
			if ( i == b)
				rra_Host[i] = 0;
			else
				rra_Host[i] = rr_Host[i*bands + b]; 
		}

		dgemm_("N", "N", &uno, &bands, &bands, &alpha, rra_Host, &uno, xx_Host, &bands, &beta, &(beta_Host[b*bands]), &uno);//no interesa llevarlo

		beta_Host[b*bands+b] = 0;
	}


	//OJO AL PARCHE HE SACAO la w fuera del bucle, voy guardando los beta en una matriz, un vector beta por cada banda
	// y al final calculo el ruido w = r - beta'*r;
	// ojo: dgemm_ hace alpha*A*B + beta*C que equivale a -1 * beta'*r + r;
	// es decir alpha = -1 beta = 1 C inicializa con la banda de la imagen (ya esta en memcpy arriba)
	// asi esta operacion se hace solo con un dgemm en vez de varios dgemm uno por cada banda, 
	// esto es para meterlo luego en GPU con cublas va todo mucho mas rapido con un solo degemm
	alpha = -1;
	beta = 1;

	magma_dsetmatrix(bands, bands, beta_Host, bands, beta_Device, size, bands, queue);
	//memcpy(noise_Device, image_Device, linessamples*bands*sizeof(double));//VV
	magma_dsetmatrix(linessamples, bands, image, linessamples, noise_Device, size, linessamples, queue);

	magma_dgemm(MagmaNoTrans, MagmaNoTrans, linessamples, bands, bands, alpha, image_Device, size, linessamples, beta_Device, size, bands, beta, noise_Device, size, linessamples, queue);



	magma_free_cpu(rr_Host);
	magma_free_cpu(rri_Host);
	magma_free_cpu(xx_Host);
	magma_free_cpu(rra_Host);
	magma_free_cpu(beta_Host);
	magma_free_cpu(ipiv_Host);
	magma_free_cpu(work_Host);

	magma_free(rr_Device);
	magma_free(beta_Device);


}


void gene_magma(double *image, int samples, int lines, int bands, int Nmax, int P_FA){

	cl_uint num_devs_returned;
	cl_context_properties properties[3];
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel_mean_pixel;

    	cl_int status;
	cl_ulong start = (cl_ulong) 0;
	cl_ulong end = (cl_ulong) 0;
	cl_event ev_mean_pixel;

	int lwork  = bands*bands, info;
	double alpha = 1,beta = 0;
	size_t size = 0;
	real_Double_t t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12;

	int linessamples = lines*samples;
	int i,j,ok=0;
	int Nmax_1 = Nmax-1;

	double k_mean_pixel = 0.0;
	size_t global_mean = 256;//ceil((double)samples*lines/256.0)*256.0;
	size_t local_mean = 256;



    	// determine number of platforms
    	cl_uint numPlatforms;
    	status = clGetPlatformIDs(0, NULL, &numPlatforms); //num_platforms returns the number of OpenCL platforms available
    	exitOnFail3(status, "number of platforms");
	

	// get platform IDs
  	cl_platform_id platformIDs[numPlatforms];
    	status = clGetPlatformIDs(numPlatforms, platformIDs, NULL); //platformsIDs returns a list of OpenCL platforms found. 
    	exitOnFail3(status, "get platform IDs");

	cl_uint numDevices;
	//cl_platform_id platformID;
        cl_device_id deviceID;
	
	//deviceSelected-> 0:CPU, 1:GPU, 2:ACCELERATOR
	int isCPU = 0, isGPU = 1, isACCEL=0;//por defecto con ClMagma vamos a usar la gpu solo
	
	// iterate over platforms
	for (i = 0; i < numPlatforms; i++){
		// determine number of devices for a platform
		status = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		//exitOnFail(status, "number of devices");
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
		               				//platformID = platformIDs[i];
		              				deviceID = deviceIDs[j];
		               			}
		               			//GPU device
		               			if (isGPU && (CL_DEVICE_TYPE_GPU & deviceType)){
							ok=1;
							//platformID = platformIDs[i];
							deviceID = deviceIDs[j];
		                		}
						//ACCELERATOR device
		               			if (isACCEL && (CL_DEVICE_TYPE_ACCELERATOR & deviceType)){
							ok=1;
							//platformID = platformIDs[i];
							deviceID = deviceIDs[j];
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

	FILE *fp;
	long filelen;
	long readlen;
	char *kernel_src;  //char string to hold kernel source
	
	fp = fopen("gene_kernel.cl","r");
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
	kernel_src[filelen]='\0';//OJO//-----------------------------------------------------------------------------------------------+1

	context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &status);//Context
	exitOnFail3( status, "clCreateContext" );
	
	// Create a command queue
	command_queue = clCreateCommandQueue(context, deviceID, CL_QUEUE_PROFILING_ENABLE, &status);
    	exitOnFail3(status, "Error: Failed to create a command queue!");
	 
	// create program object from source. 
	// kernel_src contains source read from file earlier
	program = clCreateProgramWithSource(context, 1 , (const char **) & kernel_src, NULL, &status);
	exitOnFail3(status, "Unable to create program object.");       

	status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	//exitOnFail3(status, "Build failed.");
	if (status != CL_SUCCESS)
	{
        printf("Build failed. Error Code=%d\n", status);

		size_t len;
		char buffer[2048];
		// get the build log
		clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG,sizeof(buffer), buffer, &len);
		printf("--- Build Log -- \n %s\n",buffer);
		//printf("%s\n",kernel_src);
		exit(1);
	}

	kernel_mean_pixel = clCreateKernel(program, "mean_pixel", &status);
	exitOnFail3(status, "Unable to create kernel mean_pixel object.");



	//CLMAGMA
	magma_queue_t queue;
	magma_int_t err;
	magma_init();//falla ponerle esta funcion??
	magma_print_environment();
	
	

	err = magma_queue_create( deviceID, &queue );
	if ( err != 0 ) {
		fprintf( stderr, "magma_queue_create failed: %d\n", (int) err );
		exit(-1);
	}


	//----CLMAGMA----//
	t0 = magma_sync_wtime(queue);

	//Device
	magmaDouble_ptr image_Device, noise_Device, noise_res_Device, cw_Device, cx_Device, x_Device, x_res_Device, c_Device, pixel_Device, y_Device, cov_image_Device, cov_noise_Device, img_reduced_Device, ctcw_Device, rw_small_Device;
	MALLOC_DEVICE(image_Device, double, samples*lines*bands)
	MALLOC_DEVICE(noise_Device, double, samples*lines*bands)
	MALLOC_DEVICE(noise_res_Device, double, samples*lines*bands)
	MALLOC_DEVICE(cw_Device, double, bands*bands)
	MALLOC_DEVICE(cx_Device, double, bands*bands)
	MALLOC_DEVICE(x_Device, double, samples*lines*bands)
	MALLOC_DEVICE(x_res_Device, double, samples*lines*bands)
	MALLOC_DEVICE(c_Device, double, bands*bands)
	MALLOC_DEVICE(pixel_Device, double, bands)
	MALLOC_DEVICE(y_Device, double, samples*lines*Nmax)
	MALLOC_DEVICE(cov_image_Device, double, bands*bands)
	MALLOC_DEVICE(cov_noise_Device, double, bands*bands)
	MALLOC_DEVICE(img_reduced_Device, double, Nmax*linessamples)
	MALLOC_DEVICE(ctcw_Device, double, Nmax_1*bands)
	MALLOC_DEVICE(rw_small_Device, double, Nmax_1*Nmax_1)





	//Host
	double *image_Host, *noise_Host, *cw_Host, *cx_Host, *rx_true_Host, *x_Host, *theta_Host, *work_Host, *s_Host, *c_Host, *cov_image_Host, *cov_noise_Host, *v_Host, *img_reduced_Host;
	MALLOC_HOST(image_Host, double, linessamples*bands)
	MALLOC_HOST(noise_Host, double, linessamples*bands)
	MALLOC_HOST(cw_Host, double, bands*bands)
	MALLOC_HOST(cx_Host, double, bands*bands)
	MALLOC_HOST(rx_true_Host, double, bands*bands)
	MALLOC_HOST(x_Host, double, linessamples*bands)
	MALLOC_HOST(theta_Host, double, (Nmax+1)*linessamples)
	MALLOC_HOST(work_Host, double, lwork)
	MALLOC_HOST(s_Host, double, bands)
	MALLOC_HOST(c_Host, double, bands*bands)
	MALLOC_HOST(cov_image_Host, double, bands*bands)
	MALLOC_HOST(cov_noise_Host, double, bands*bands)
	MALLOC_HOST(v_Host, double, bands*bands)
	MALLOC_HOST(img_reduced_Host, double, Nmax*linessamples)



//---------------------------------------------------------------------------------------------
//                                     Pruebas
	/*int filas = 3, columnas = 3;
	double *A_h, *B_h;
	cl_mem A_k, B_k;
	magma_malloc(&A_k, filas*columnas*sizeof(double));
	magma_malloc(&B_k, filas*columnas*sizeof(double));
	magma_malloc_cpu((void**) &A_h, filas*columnas*sizeof(double));
	magma_malloc_cpu((void**) &B_h, filas*columnas*sizeof(double));
	for(int z = 0; z < filas*columnas; z++) A_h[z] = z+1;
	for(int z = 0; z < filas*columnas; z++) printf("%f ",A_h[z]); printf("\n");

	magma_dsetmatrix(filas, columnas, A_h, filas, A_k, size, filas, queue);
	magma_dgemm(MagmaNoTrans, MagmaTrans, filas, filas, columnas, alpha, A_k, size, filas, A_k, size, filas, beta, B_k, size, filas, queue);
	magma_dgetmatrix(filas,columnas, B_k, size, filas, B_h, filas, queue);

	for(int z = 0; z < filas*columnas; z++) printf("%f ",B_h[z]); printf("\n");//hasta aqui hemos usado un cl_mem como magma pointer -> ok!
//--
	cl_kernel k_prueba;
	k_prueba = clCreateKernel(program, "prueba", &status);
	
	status = clSetKernelArg(k_prueba, 0, sizeof(cl_mem), &A_k);
	exitOnFail3(status, "Unable to set kernel prueba arguments.");

	size_t global_prueba = 9;
	status = clEnqueueNDRangeKernel(command_queue, k_prueba, 1, NULL, &global_prueba, NULL, 0, NULL, NULL);
	exitOnFail3(status, "Launch OpenCL prueba kernel");

	magma_dgetmatrix(filas,columnas, A_k, size, filas, A_h, filas, queue);
	for(int z = 0; z < filas*columnas; z++) printf("%f ",A_h[z]); printf("\n");*/




//---------------------------------------------------------------------------------------------


	//kernel
	cl_mem image_kernel = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * linessamples * bands, image, &status);
	exitOnFail3(status, "create buffer noise");


	status = clSetKernelArg(kernel_mean_pixel, 0, sizeof(cl_mem), &image_kernel);
	status |= clSetKernelArg(kernel_mean_pixel, 1, sizeof(cl_int), &linessamples);
	status |= clSetKernelArg(kernel_mean_pixel, 2, sizeof(cl_int), &bands);
    	status |= clSetKernelArg(kernel_mean_pixel, 3, sizeof(cl_double)*local_mean, NULL);//localsize
	exitOnFail3(status, "Unable to set kernel mean_pixel arguments.");


	//---------------------------------------//


	magma_dsetmatrix(samples*lines, bands, image, samples*lines, image_Device, size, samples*lines, queue);
	//magma_dsetmatrix(samples*lines, bands, image, samples*lines, x_Device, size, samples*lines, queue);//??????????????
	t1 = magma_sync_wtime(queue);

	//---------------------------------------//(noise reduction)

	est_noise(image, image_Device, linessamples, bands, noise_Device, queue);
	
	//perogrullada llevarlo a host para luego traerlo a device de nuevo
	magma_dgetmatrix(linessamples, bands, noise_Device, size, linessamples, noise_Host, linessamples, queue);//comprobar que es necesario(donde se usa noise host)

	t2 = magma_sync_wtime(queue);



	//---------------------------------------//(covarianza)

	//(covarianza de la imagen)
	status = clEnqueueNDRangeKernel(command_queue, kernel_mean_pixel, 1, NULL, &global_mean, &local_mean, 0, NULL, &ev_mean_pixel);//covarianza sobre imagen
	exitOnFail3(status, "Launch OpenCL mean_pixel kernel");
	start = (cl_ulong) 0;
	end = (cl_ulong) 0;
	clWaitForEvents(1,&ev_mean_pixel);
	status = clGetEventProfilingInfo(ev_mean_pixel, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	exitOnFail3(status, "Profiling kernel endmember - start");
	status = clGetEventProfilingInfo(ev_mean_pixel, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	exitOnFail3(status, "Profiling kernel endmember - end");
	clReleaseEvent(ev_mean_pixel);
	k_mean_pixel+=(end-start)*1.0e-9;


//--llevarlo a host para traerlo a device
	status = clEnqueueReadBuffer(command_queue, image_kernel, CL_TRUE, 0, sizeof(double) * linessamples * bands, image_Host, 0, NULL, NULL);//cozarianza de noise
	exitOnFail3(status, "Error enqueuing read buffer command.");//medir tiempo tambien??

	magma_dsetmatrix(linessamples, bands, image_Host, linessamples, image_Device, size, linessamples, queue);
	magma_dgemm(MagmaTrans, MagmaNoTrans, bands, bands, linessamples, alpha, image_Device, size, linessamples, image_Device, size, linessamples, beta, cov_image_Device, size, bands, queue);
//---

	//(covarianza de noise) (pasamos por host para traer un array que ya estaba en device...)
	cl_mem noise_kernel = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * linessamples * bands, noise_Host, &status);
	exitOnFail3(status, "create buffer noise");
	status = clSetKernelArg(kernel_mean_pixel, 0, sizeof(cl_mem), &noise_kernel);//con esto funciona pero es traerlo desde Host...
	

	status |= clEnqueueNDRangeKernel(command_queue, kernel_mean_pixel, 1, NULL, &global_mean, &local_mean, 0, NULL, &ev_mean_pixel);//covarianza sobre imagen
	exitOnFail3(status, "Launch OpenCL mean_pixel kernel 2");
	start = (cl_ulong) 0;
	end = (cl_ulong) 0;
	clWaitForEvents(1,&ev_mean_pixel);
	status = clGetEventProfilingInfo(ev_mean_pixel, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	exitOnFail3(status, "Profiling kernel endmember - start");
	status = clGetEventProfilingInfo(ev_mean_pixel, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	exitOnFail3(status, "Profiling kernel endmember - end");
	clReleaseEvent(ev_mean_pixel);
	k_mean_pixel+=(end-start)*1.0e-9;

//--llevarlo a host para traerlo a device
	status = clEnqueueReadBuffer(command_queue, noise_kernel, CL_TRUE, 0, sizeof(double) * linessamples * bands, noise_Host, 0, NULL, NULL);//cozarianza de img
	exitOnFail3(status, "Error enqueuing read buffer command.");//medir tiempo tambien??

	magma_dsetmatrix(linessamples, bands, noise_Host, linessamples, noise_Device, size, linessamples, queue);
	magma_dgemm(MagmaTrans, MagmaNoTrans, bands, bands, linessamples, alpha, noise_Device, size, linessamples, noise_Device, size, linessamples, beta, cov_noise_Device, size, bands, queue);
//---



	t3 = magma_sync_wtime(queue);

	//---------------------------------------//

	/********************************************************************************************************************************************/
	//Several things	
	//gettimeofday(&t4,NULL);
	/********************************************************************************************************************************************/

	magma_dgetmatrix(bands, bands, cov_image_Device, size, bands, cov_image_Host, bands, queue);
	magma_dgetmatrix(bands, bands, cov_noise_Device, size, bands, cov_noise_Host, bands, queue);

	// Calculamos Rx_true quitando a la covarianza de la imagen la covarianza del ruido.
	for (i = 0; i < ( bands*bands); i++){
		rx_true_Host[i] = cov_image_Host[i] - cov_noise_Host[i];
		cov_noise_Host[i] /= linessamples;
	}
	magma_dsetmatrix(bands, bands, cov_noise_Host, bands, cov_noise_Device, size, bands, queue);
	

	
	magma_dgesvd(MagmaNoVec, MagmaAllVec, bands, bands, rx_true_Host, bands, s_Host, v_Host, bands, c_Host, bands, work_Host, lwork, queue, &info);
	magma_dsetmatrix(bands, bands, c_Host, bands, c_Device, size, bands, queue);


	magma_dgemm(MagmaNoTrans, MagmaTrans, linessamples, Nmax_1, bands, alpha, image_Device, size, linessamples, c_Device, size, bands, beta, img_reduced_Device, size, linessamples, queue);
	magma_dgetmatrix(Nmax, linessamples, img_reduced_Device, size, Nmax, img_reduced_Host, Nmax, queue);
	for (i = 0; i < linessamples; i++){
		img_reduced_Host[linessamples*Nmax_1 +i] = 1;
	}
//----

//----
	//C'*Cw 	dgemm_("N", "N", &Nmax_1, &bands, &bands, &alpha, C, &bands, Cw, &bands, &beta, CtCw, &Nmax_1);
	magma_dgemm(MagmaNoTrans, MagmaNoTrans, Nmax_1, bands, bands, alpha, c_Device, size, bands, cov_noise_Device, size, bands, beta, ctcw_Device, size, Nmax_1, queue);

	//C'*Cw*C
	magma_dgemm(MagmaNoTrans, MagmaTrans, Nmax_1, Nmax_1, bands, alpha, ctcw_Device, size, Nmax_1, c_Device, size, bands, beta, rw_small_Device, size, Nmax_1, queue);



/*
	int *ipiv = (int*)malloc( (Nmax-1)*sizeof(int));
	lwork = (Nmax_1)*(Nmax_1);
	free(work);
	work = (double*)malloc(lwork*sizeof(double));


	double* invRsmall = (double*)malloc((Nmax_1)*(Nmax_1)*sizeof(double));
	memcpy (invRsmall, Rw_small,(Nmax_1)*(Nmax_1)*sizeof(double));
	dgetrf_(&Nmax_1,&Nmax_1,invRsmall,&Nmax_1,ipiv,&info);
	dgetri_(&Nmax_1,invRsmall,&Nmax_1,ipiv,work,&lwork,&info);
*/



	t4 = magma_sync_wtime(queue);

	//---------------------------------------//
	//---------------------------------------//
	//---------------------------------------//
	//---------------------------------------//
	//---------------------------------------//
	//---------------------------------------//
	//---------------------------------------//
	//---------------------------------------//
	//---------------------------------------//

	printf("Init: %f\n",t1-t0);
	printf("Noise reduction: %f\n",t2-t1);
	printf("Covarianza: %f\n",t3-t2);
	printf("stuff: %f\n",t4-t3);

//hacer los free MAGMA

	clFinish(command_queue);

	clReleaseProgram(program);
	clReleaseKernel(kernel_mean_pixel);
	clReleaseCommandQueue(command_queue);
    	clReleaseContext(context);

	free(kernel_src);


}


void exitOnFail3(cl_int status, const char* message){
	if (CL_SUCCESS != status){
		printf("error: %s\n", message);
		printf("error: %d\n", status);
		
		switch (status) {

        case CL_SUCCESS :
            printf(" CL_SUCCESS "); break;
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
            printf("UNKNOWN ERROR");

    }
		exit(-1);
	}
}
