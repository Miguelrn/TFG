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


void gene_magma(double *image, int samples, int lines, int bands, int Nmax, int P_FA, cl_command_queue command_queue, cl_context context, cl_device_id deviceID){


	cl_program program;
	cl_kernel kernel_mean_pixel;

    	cl_int status;
	cl_ulong start = (cl_ulong) 0;
	cl_ulong end = (cl_ulong) 0;
	cl_event ev_mean_pixel;

	int lwork  = bands*bands, info;
	double alpha = 1.0,beta = 0.0;
	size_t size = 0;
	real_Double_t t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12;

	int linessamples = lines*samples;
	int i,j,ok=0;
	int Nmax_1 = Nmax-1;

	double k_mean_pixel = 0.0;
	size_t global_mean = 256;//ceil((double)samples*lines/256.0)*256.0;
	size_t local_mean = 256;


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

	 
	// create program object from source. 
	// kernel_src contains source read from file earlier
	program = clCreateProgramWithSource(context, 1 , (const char **) & kernel_src, NULL, &status);
	exitOnFail(status, "Unable to create program object.");       

	status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	//exitOnFail(status, "Unable to build program.");
	if (status != CL_SUCCESS){
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
	exitOnFail(status, "Unable to create kernel mean_pixel object.");



	//CLMAGMA
	magma_queue_t queue;
	magma_int_t err;
	magma_init();//falla ponerle esta funcion??
	magma_print_environment();
	
	

	err = magma_queue_create(deviceID, &queue);
	if ( err != 0 ) {
		fprintf( stderr, "magma_queue_create failed: %d\n", (int) err );
		exit(-1);
	}


	//----CLMAGMA----//
	t0 = magma_sync_wtime(queue);

	//Device
	magmaDouble_ptr image_Device, noise_Device, c_Device, cov_image_Device, cov_noise_Device, img_reduced_Device, ctcw_Device, rw_small_Device, work_Device;
	magma_int_t ldwork, ldda;
	ldwork = Nmax_1 * magma_get_dgetri_nb(Nmax_1);
	ldda = ((Nmax_1+31)/32)*32;

	MALLOC_DEVICE(image_Device, double, samples*lines*bands)
	MALLOC_DEVICE(noise_Device, double, samples*lines*bands)
	MALLOC_DEVICE(c_Device, double, bands*bands)
	MALLOC_DEVICE(cov_image_Device, double, bands*bands)
	MALLOC_DEVICE(cov_noise_Device, double, bands*bands)
	MALLOC_DEVICE(img_reduced_Device, double, Nmax*linessamples)
	MALLOC_DEVICE(ctcw_Device, double, Nmax_1*bands)
	MALLOC_DEVICE(rw_small_Device, double, ldda*Nmax_1)
	MALLOC_DEVICE(work_Device, double, ldwork)






	//Host
	double *image_Host, *noise_Host, *rx_true_Host, *work_Host, *s_Host, *c_Host, *cov_image_Host, *cov_noise_Host, *v_Host, *img_reduced_Host;
	magma_int_t *ipiv_Host;
	MALLOC_HOST(image_Host, double, linessamples*bands)
	MALLOC_HOST(noise_Host, double, linessamples*bands)
	MALLOC_HOST(rx_true_Host, double, bands*bands)
	MALLOC_HOST(work_Host, double, lwork)
	MALLOC_HOST(s_Host, double, bands)
	MALLOC_HOST(c_Host, double, bands*bands)
	MALLOC_HOST(cov_image_Host, double, bands*bands)
	MALLOC_HOST(cov_noise_Host, double, bands*bands)
	MALLOC_HOST(v_Host, double, bands*bands)
	MALLOC_HOST(img_reduced_Host, double, Nmax*linessamples)
	MALLOC_HOST(ipiv_Host, magma_int_t, Nmax_1)



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
	exitOnFail(status, "Unable to set kernel prueba arguments.");

	size_t global_prueba = 9;
	status = clEnqueueNDRangeKernel(command_queue, k_prueba, 1, NULL, &global_prueba, NULL, 0, NULL, NULL);
	exitOnFail(status, "Launch OpenCL prueba kernel");

	magma_dgetmatrix(filas,columnas, A_k, size, filas, A_h, filas, queue);
	for(int z = 0; z < filas*columnas; z++) printf("%f ",A_h[z]); printf("\n");*/




//---------------------------------------------------------------------------------------------

	//kernel
	cl_mem image_kernel = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * linessamples * bands, image, &status);
	exitOnFail(status, "create buffer noise");


	status = clSetKernelArg(kernel_mean_pixel, 0, sizeof(cl_mem), &image_kernel);
	status |= clSetKernelArg(kernel_mean_pixel, 1, sizeof(cl_int), &linessamples);
	status |= clSetKernelArg(kernel_mean_pixel, 2, sizeof(cl_int), &bands);
    	status |= clSetKernelArg(kernel_mean_pixel, 3, sizeof(cl_double)*local_mean, NULL);//localsize
	exitOnFail(status, "Unable to set kernel mean_pixel arguments.");


	//---------------------------------------//


	magma_dsetmatrix(samples*lines, bands, image, samples*lines, image_Device, size, samples*lines, queue);
	t1 = magma_sync_wtime(queue);

	//---------------------------------------//(noise reduction)

	est_noise(image, image_Device, linessamples, bands, noise_Device, queue);
	
	//perogrullada llevarlo a host para luego traerlo a device de nuevo
	magma_dgetmatrix(linessamples, bands, noise_Device, size, linessamples, noise_Host, linessamples, queue);//comprobar que es necesario(donde se usa noise host)

	t2 = magma_sync_wtime(queue);



	//---------------------------------------//(covarianza)

	//(covarianza de la imagen)
	status = clEnqueueNDRangeKernel(command_queue, kernel_mean_pixel, 1, NULL, &global_mean, &local_mean, 0, NULL, &ev_mean_pixel);//covarianza sobre imagen
	exitOnFail(status, "Launch OpenCL mean_pixel kernel");
	start = (cl_ulong) 0;
	end = (cl_ulong) 0;
	clWaitForEvents(1,&ev_mean_pixel);
	status = clGetEventProfilingInfo(ev_mean_pixel, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	exitOnFail(status, "Profiling kernel endmember - start");
	status = clGetEventProfilingInfo(ev_mean_pixel, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	exitOnFail(status, "Profiling kernel endmember - end");
	clReleaseEvent(ev_mean_pixel);
	k_mean_pixel+=(end-start)*1.0e-9;


//--llevarlo a host para traerlo a device
	status = clEnqueueReadBuffer(command_queue, image_kernel, CL_TRUE, 0, sizeof(double) * linessamples * bands, image_Host, 0, NULL, NULL);//cozarianza de noise
	exitOnFail(status, "Error enqueuing read buffer command.");//medir tiempo tambien??

	magma_dsetmatrix(linessamples, bands, image_Host, linessamples, image_Device, size, linessamples, queue);
	magma_dgemm(MagmaTrans, MagmaNoTrans, bands, bands, linessamples, alpha, image_Device, size, linessamples, image_Device, size, linessamples, beta, cov_image_Device, size, bands, queue);
//---

	//(covarianza de noise) (pasamos por host para traer un array que ya estaba en device...)
	cl_mem noise_kernel = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * linessamples * bands, noise_Host, &status);
	exitOnFail(status, "create buffer noise");
	status = clSetKernelArg(kernel_mean_pixel, 0, sizeof(cl_mem), &noise_kernel);//con esto funciona pero es traerlo desde Host...
	

	status |= clEnqueueNDRangeKernel(command_queue, kernel_mean_pixel, 1, NULL, &global_mean, &local_mean, 0, NULL, &ev_mean_pixel);//covarianza sobre imagen
	exitOnFail(status, "Launch OpenCL mean_pixel kernel 2");
	start = (cl_ulong) 0;
	end = (cl_ulong) 0;
	clWaitForEvents(1,&ev_mean_pixel);
	status = clGetEventProfilingInfo(ev_mean_pixel, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	exitOnFail(status, "Profiling kernel endmember - start");
	status = clGetEventProfilingInfo(ev_mean_pixel, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	exitOnFail(status, "Profiling kernel endmember - end");
	clReleaseEvent(ev_mean_pixel);
	k_mean_pixel+=(end-start)*1.0e-9;

//--llevarlo a host para traerlo a device
	status = clEnqueueReadBuffer(command_queue, noise_kernel, CL_TRUE, 0, sizeof(double) * linessamples * bands, noise_Host, 0, NULL, NULL);//cozarianza de img
	exitOnFail(status, "Error enqueuing read buffer command.");//medir tiempo tambien??

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
		img_reduced_Host[linessamples*Nmax_1 +i] = 1.0;
	}


	//C'*Cw 
	magma_dgemm(MagmaNoTrans, MagmaNoTrans, Nmax_1, bands, bands, alpha, c_Device, size, bands, cov_noise_Device, size, bands, beta, ctcw_Device, size, Nmax_1, queue);

	//C'*Cw*C 
	magma_dgemm(MagmaNoTrans, MagmaTrans, Nmax_1, Nmax_1, bands, alpha, ctcw_Device, size, Nmax_1, c_Device, size, bands, beta, rw_small_Device, size, ldda, queue);


	magma_dgetrf_gpu(Nmax_1, Nmax_1, rw_small_Device, size, ldda, ipiv_Host, queue, &info);
	magma_dgetri_gpu(Nmax_1, rw_small_Device, size, ldda, ipiv_Host, work_Device, size, ldwork, &queue, &info);



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


	free(kernel_src);


}



