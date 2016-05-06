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
	cl_kernel kernel_max_bright;
	cl_kernel kernel_max_bright_reduce;
	cl_kernel kernel_pixel_projections;

    	cl_int status;
	cl_ulong start = (cl_ulong) 0;
	cl_ulong end = (cl_ulong) 0;
	cl_event ev_mean_pixel;
	cl_event ev_max_bright;

	int lwork  = bands*bands, info;
	double alpha = 1.0,beta = 0.0;
	size_t size = 0;
	real_Double_t t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12;

	int linessamples = lines*samples;
	int i,j,ok=0;
	int Nmax_1 = Nmax-1;
	int pos_abs;
	double max_bright = 0.0;
	double r_to_inf = -1;

	double k_mean_pixel = 0.0, k_max_bright = 0.0;
	size_t global_mean = 256;//ceil((double)samples*lines/256.0)*256.0;
	size_t local_mean = 256;
	size_t global_bright = linessamples;
	size_t local_projections = 256;//por ejemplo...
	size_t global_projections = ceil(linessamples/local_projections)*local_projections;

	int totalProjections = global_projections/local_projections;
	double *projections = (double*) malloc (totalProjections * sizeof(double));
	int *indice = (int*) malloc (totalProjections * sizeof(int));
	int positions[Nmax][2];

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
	exitOnFail(status, "Unable to build program.");

	kernel_mean_pixel = clCreateKernel(program, "mean_pixel", &status);
	exitOnFail(status, "Unable to create kernel mean_pixel object.");

	kernel_max_bright = clCreateKernel(program, "max_bright", &status);
	exitOnFail(status, "Unable to create kernel max_bright object.");

	kernel_max_bright_reduce = clCreateKernel(program, "max_bright_reduce", &status);
	exitOnFail(status, "Unable to create kernel max_bright_reduce object.");

	kernel_pixel_projections = clCreateKernel(program, "pixel_projections", &status);
	exitOnFail(status, "Unable to create kernel kernel_pixel_projection object.");



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
	double *image_Host, *noise_Host, *rx_true_Host, *work_Host, *s_Host, *c_Host, *cov_image_Host, *cov_noise_Host, *v_Host, *img_reduced_Host, *umatrix_Host;
	double *mul_umatrix_Host, *mul_umatrix_inv_Host, *umatrix_aux_Host, *proymatrix_Host, *endmember_Host, *theta_Host, *rw_small_Host;
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
	MALLOC_HOST(umatrix_Host, double, Nmax*Nmax)
	MALLOC_HOST(mul_umatrix_Host, double, Nmax*Nmax)
	MALLOC_HOST(mul_umatrix_inv_Host, double, Nmax*Nmax)
	MALLOC_HOST(umatrix_aux_Host, double, Nmax*Nmax)
	MALLOC_HOST(proymatrix_Host, double, Nmax*Nmax)
	MALLOC_HOST(endmember_Host, double, Nmax)
	MALLOC_HOST(theta_Host, double, (Nmax+1)*linessamples)
	MALLOC_HOST(rw_small_Host, double, Nmax_1*Nmax_1)



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
	exitOnFail(status, "Profiling kernel mean_pixel - start");
	status = clGetEventProfilingInfo(ev_mean_pixel, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	exitOnFail(status, "Profiling kernel mean_pixel - end");
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
	exitOnFail(status, "Profiling kernel mean_pixel - start");
	status = clGetEventProfilingInfo(ev_mean_pixel, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	exitOnFail(status, "Profiling kernel mean_pixel - end");
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
	magma_dgetmatrix(Nmax, linessamples, img_reduced_Device, size, Nmax, img_reduced_Host, Nmax, queue);//ya lo tengo en device...
	/*for (i = 0; i < linessamples; i++){//hacer kernel de esto?
		img_reduced_Host[linessamples*Nmax_1 +i] = 1.0;
	}*/


	//C'*Cw 
	magma_dgemm(MagmaNoTrans, MagmaNoTrans, Nmax_1, bands, bands, alpha, c_Device, size, bands, cov_noise_Device, size, bands, beta, ctcw_Device, size, Nmax_1, queue);

	//C'*Cw*C 
	magma_dgemm(MagmaNoTrans, MagmaTrans, Nmax_1, Nmax_1, bands, alpha, ctcw_Device, size, Nmax_1, c_Device, size, bands, beta, rw_small_Device, size, ldda, queue);


	magma_dgetrf_gpu(Nmax_1, Nmax_1, rw_small_Device, size, ldda, ipiv_Host, queue, &info);
	magma_dgetri_gpu(Nmax_1, rw_small_Device, size, ldda, ipiv_Host, work_Device, size, ldwork, &queue, &info);

	magma_dgetmatrix(Nmax_1, Nmax_1, rw_small_Device, size, ldda, rw_small_Host, Nmax_1, queue);//lo usa np_test


	/*double max = 0.0, min = 0.0;
	for(i = 0; i < Nmax_1*Nmax_1; i++){
		if(max < rw_small_Host[i]){
			max = rw_small_Host[i];
		}
		if(min > rw_small_Host[i])
			min = rw_small_Host[i];
		
	}printf("%f - %f\n",max,min);*/

	

	t4 = magma_sync_wtime(queue);

	//---------------------------------------//
	// ahora empieza el ATGP.

	// The brightest pixel is calculated
	//pos_abs = Get_pixel_max_bright(image_vector, num_bands,lines_samples);//hacerlo kernel
	//ya lo tengo en device como img_reduced_Device.... pero no puedo usarlo z.z
	cl_mem img_reduced_kernel = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double)*linessamples*Nmax, img_reduced_Host, &status);
	exitOnFail(status, "create buffer img_reduced");
	cl_mem img_max_bright = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*linessamples, NULL, &status);
	exitOnFail(status, "create buffer img_max_bright");
	cl_mem img_projections = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*totalProjections, NULL, &status);
	exitOnFail(status, "create buffer img_projections");
	cl_mem indice_projections = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int)*totalProjections, NULL, &status);
	exitOnFail(status, "create buffer indice_projections");
	cl_mem proymatrix_kernel = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double)*Nmax*Nmax, NULL, &status);
	exitOnFail(status, "create buffer proymatrix_kernel");
	

	status = clSetKernelArg(kernel_max_bright, 0, sizeof(cl_mem), &img_reduced_kernel);
	status |= clSetKernelArg(kernel_max_bright, 1, sizeof(cl_int), &linessamples);
	status |= clSetKernelArg(kernel_max_bright, 2, sizeof(cl_int), &Nmax);
    	status |= clSetKernelArg(kernel_max_bright, 3, sizeof(cl_mem), &img_max_bright);
	exitOnFail(status, "Unable to set kernel max_bright arguments.");

	status = clSetKernelArg(kernel_max_bright_reduce, 0, sizeof(cl_mem), &img_max_bright);
	status |= clSetKernelArg(kernel_max_bright_reduce, 1, sizeof(cl_int), &linessamples);
	status |= clSetKernelArg(kernel_max_bright_reduce, 2, sizeof(cl_mem), &indice_projections);
    	status |= clSetKernelArg(kernel_max_bright_reduce, 3, sizeof(cl_mem), &img_projections);
	exitOnFail(status, "Unable to set kernel max_bright arguments.");

	status = clSetKernelArg(kernel_pixel_projections, 0, sizeof(cl_mem), &img_reduced_kernel);
	status = clSetKernelArg(kernel_pixel_projections, 1, sizeof(cl_mem), &proymatrix_kernel);
	status |= clSetKernelArg(kernel_pixel_projections, 2, sizeof(cl_int), &linessamples);
	status |= clSetKernelArg(kernel_pixel_projections, 3, sizeof(cl_int), &Nmax);
    	status |= clSetKernelArg(kernel_pixel_projections, 4, sizeof(cl_mem), &img_max_bright);
	exitOnFail(status, "Unable to set kernel pixel projections arguments.");

	status = clEnqueueNDRangeKernel(command_queue, kernel_max_bright, 1, NULL, &global_bright, NULL, 0, NULL, &ev_max_bright);//max_bright
	exitOnFail(status, "Launch OpenCL max_bright kernel");
	start = (cl_ulong) 0;
	end = (cl_ulong) 0;
	clWaitForEvents(1,&ev_max_bright);
	status = clGetEventProfilingInfo(ev_max_bright, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	exitOnFail(status, "Profiling kernel max_bright - start");
	status = clGetEventProfilingInfo(ev_max_bright, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	exitOnFail(status, "Profiling kernel max_bright - end");
	clReleaseEvent(ev_max_bright);
	k_max_bright+=(end-start)*1.0e-9;


	status = clEnqueueNDRangeKernel(command_queue, kernel_max_bright_reduce, 1, NULL, &global_projections, &local_projections, 0, NULL, &ev_max_bright);
	exitOnFail(status, "Launch OpenCL max_bright_reduction kernel");
	start = (cl_ulong) 0;
	end = (cl_ulong) 0;
	clWaitForEvents(1,&ev_max_bright);
	status = clGetEventProfilingInfo(ev_max_bright, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	exitOnFail(status, "Profiling kernel max_bright_reduction - start");
	status = clGetEventProfilingInfo(ev_max_bright, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	exitOnFail(status, "Profiling kernel max_bright_reduction - end");
	clReleaseEvent(ev_max_bright);
	k_max_bright+=(end-start)*1.0e-9;


	clEnqueueReadBuffer(command_queue, indice_projections, CL_TRUE, 0, sizeof(int) * totalProjections, indice, 0, NULL, NULL);
	clEnqueueReadBuffer(command_queue, img_projections, CL_TRUE, 0, sizeof(double) * totalProjections, projections, 0, NULL, NULL);
	
	
	for(i = 0; i < totalProjections; i++){
		if(max_bright < projections[i]){
			pos_abs = indice[i];
			max_bright = projections[i];
		}
	}
	//printf("%d = (%d - %d)\n",pos_abs, pos_abs / samples, pos_abs % samples);//funciona !!!!

	positions[0][0] = pos_abs / samples;
	positions[0][1] = pos_abs % samples;
printf("%d = (%d - %d)\n",pos_abs, pos_abs / samples, pos_abs % samples);//funciona!!

	for (i = 0; i < Nmax_1; i++){
		umatrix_Host[i]= img_reduced_Host[pos_abs+(i*linessamples)]; //max bright image_vector
	}
	umatrix_Host[Nmax_1] = 1;

	i = 1;

	t5 = magma_sync_wtime(queue);

	//---------------------------------------//
	// Launch the ATGP algorithm to find i-1 targets (the first target is already available)
	while((r_to_inf <= P_FA) && (i < Nmax)){

		
		UtxU(umatrix_Host, mul_umatrix_Host, i, Nmax);
		GaussSeidel_seq(mul_umatrix_Host, mul_umatrix_inv_Host, i);
		Uxinv(umatrix_Host, mul_umatrix_inv_Host, umatrix_aux_Host, i, Nmax);
 		AnsxUt(umatrix_aux_Host, umatrix_Host, proymatrix_Host, i, Nmax);
		SustractIdentity(proymatrix_Host, Nmax);

		//envio del proymatrix lanzamiento de los dos kernel y lectura y reduccion final
		status = clEnqueueWriteBuffer(command_queue, proymatrix_kernel, CL_TRUE, 0, sizeof(double) * Nmax * Nmax , proymatrix_Host, 0, NULL, NULL);
		exitOnFail(status, "Write proymatrix buffer");

		status = clEnqueueNDRangeKernel(command_queue, kernel_pixel_projections, 1, NULL, &global_bright, NULL, 0, NULL, &ev_max_bright);//max_bright
		exitOnFail(status, "Launch OpenCL pixel_projections kernel");
		start = (cl_ulong) 0;
		end = (cl_ulong) 0;
		clWaitForEvents(1,&ev_max_bright);
		status = clGetEventProfilingInfo(ev_max_bright, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
		exitOnFail(status, "Profiling kernel max_bright - start");
		status = clGetEventProfilingInfo(ev_max_bright, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
		exitOnFail(status, "Profiling kernel max_bright - end");
		clReleaseEvent(ev_max_bright);
		k_max_bright+=(end-start)*1.0e-9;

		status = clEnqueueNDRangeKernel(command_queue, kernel_max_bright_reduce, 1, NULL, &global_projections, &local_projections, 0, NULL, &ev_max_bright);
		exitOnFail(status, "Launch OpenCL max_bright_reduction kernel");
		start = (cl_ulong) 0;
		end = (cl_ulong) 0;
		clWaitForEvents(1,&ev_max_bright);
		status = clGetEventProfilingInfo(ev_max_bright, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
		exitOnFail(status, "Profiling kernel max_bright_reduction - start");
		status = clGetEventProfilingInfo(ev_max_bright, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
		exitOnFail(status, "Profiling kernel max_bright_reduction - end");
		clReleaseEvent(ev_max_bright);
		k_max_bright+=(end-start)*1.0e-9;

		clEnqueueReadBuffer(command_queue, indice_projections, CL_TRUE, 0, sizeof(int) * totalProjections, indice, 0, NULL, NULL);
		clEnqueueReadBuffer(command_queue, img_projections, CL_TRUE, 0, sizeof(double) * totalProjections, projections, 0, NULL, NULL);
	
		max_bright = 0.0;
		for(j = 0; j < totalProjections; j++){
			if(max_bright < projections[j]){
				pos_abs = indice[j];
				max_bright = projections[j];
			}
		}
		
		

		if (i > 2){
			
			for(j = 0; j < Nmax; j++){
				endmember_Host[j] = img_reduced_Host[pos_abs+(j*linessamples)];
			}

			//Obtener theta
			lsu_gpu_m(umatrix_Host, endmember_Host, deviceID, Nmax, i, 1, 1, NULL, theta_Host);


			//calcular el r_to_inf con el test de newman person
			r_to_inf = GENE_NP_test(theta_Host, Nmax, i, umatrix_Host, endmember_Host, rw_small_Host);
		
		}
		if ( r_to_inf <= P_FA){
			
			positions[i][0]= (pos_abs / samples);
			positions[i][1]= (pos_abs % samples);	
		
			//printf("%d = (%d - %d)\n",pos_abs, pos_abs / samples, pos_abs % samples);//funciona!!

			if(i < Nmax_1){
				for(j = 0; j < Nmax_1; j++)
					umatrix_Host[j+i*Nmax] = img_reduced_Host[pos_abs+j*linessamples];
				umatrix_Host[Nmax_1+i*bands] = 1;
				
			}
			i++;
		}
		

	}

	t5 = magma_sync_wtime(queue);

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
	printf("ATGP %f\n",t5-t4);

	for (j = 0; j < i; j++){
		printf("Pos(%2d) = [%d,%d] \n",j,positions[j][0], positions[j][1]);
	}

//hacer los free MAGMA

	clFinish(command_queue);

	clReleaseProgram(program);
	clReleaseKernel(kernel_mean_pixel);


	free(kernel_src);


}


void UtxU(double *umatrix, double *mul_umatrix, int iter,int num_bands){
	int j, k, div, aux1, aux2, modulus;
	double valor;
	
	for(j = 0; j < iter*iter; j++){
		valor=0.0;
		div=(int)(j/iter);//para que es el cast?
		aux1=j/iter;
		aux2=iter*aux1;
		modulus=j-aux2;
		for(k=0;k<num_bands;k+=1)
			valor+=umatrix[k+num_bands*div]*umatrix[k+num_bands*modulus];
		mul_umatrix[j]=valor;
	}	
}

int GaussSeidel_seq (double *matrix, double *inv_matrix, int size){
	int i;
	double *ext_matrix;
	double *ext_matrix_aux;
	int size_x2 = 2 * size;

	// Reservamos memoria para la matriz y la identidad (extendida)
	ext_matrix = (double *) malloc (2 * size * size * sizeof (double));

	if (ext_matrix == 0) return -1;

	// Almacena la matriz con la matriz identidad
	CreateExtMatrix (matrix, ext_matrix, size, size_x2);

	// 1. Diagonal inferior izquierda a cero
	ProcessLowerLeftDiag (ext_matrix, size, size_x2);
	// 2. Diagonal superior derecha a cero
	ProcessUpperRightDiag (ext_matrix, size, size_x2);

	// Normalizar la diagonal para conseguir la matriz identidad a la izquierda
	ProcessDiag (ext_matrix, size, size_x2);

	// Guardamos la matriz inversa
	ext_matrix_aux = ext_matrix;
	for (i = 0; i < size; i++){
		memcpy (inv_matrix, ext_matrix_aux + size, size * sizeof (double));
		inv_matrix += size;
		ext_matrix_aux += size_x2;
	}

	free (ext_matrix);

	return 0;
}

void CreateExtMatrix (double* src, double* dest, int size, int size_x2){
	int i;

	for (i = 0; i < size; i++){
		memcpy (dest, src, size * sizeof (double));
		memset (dest + size, 0, size * sizeof (double));
		dest[i + size] = 1;
		dest += size_x2;
		src += size;
	}
}

int ProcessLowerLeftDiag (double *matrix, int size, int size_x2){
	int i, j, k;

	double *last_row_piv = matrix;

	
	for (i = 0; i < size; i++){//columns
		for (j = i + 1; j < size; j++){//rows
			double pivot_aux;
			double *row_aux = matrix + (j * size_x2);//WTF

			if (last_row_piv[i] == 0){
				pivot_aux = 0;
			}
			else {
				pivot_aux = row_aux[i] / last_row_piv[i];
			}

			for (k = 0; k < size_x2; k++){
				row_aux[k] -= last_row_piv[k] * pivot_aux;
			}
		}
		last_row_piv += size_x2;
	}
	return 0;
}

int ProcessUpperRightDiag (double *matrix, int size, int size_x2){
	int i, j, k;

	// Puntero a la última fila
	double *last_row_piv = matrix + size_x2 * (size - 1);


	for (i = (size - 1); i > 0; i--){//columns

		for (j = 0; j < i; j++){//rows
			double pivot_aux;
			double *row_aux = matrix + (j * size_x2);

			if (last_row_piv[i] == 0){
				pivot_aux = 0;
			}
			else {
				pivot_aux = row_aux[i] / last_row_piv[i];
			}

			// Bucle que recorre columna y resta el pivote
			for (k = 0; k < size_x2; k++){
				row_aux[k] -= last_row_piv[k] * pivot_aux;
			}
		}
		last_row_piv -= size_x2;
	}
	return 0;
}




void ProcessDiag (double *matrix, int size, int size_x2){
	int i, j;

	for (i = 0; i < size; i++){
		double val = matrix[i];

		for (j = 0; j < size_x2; j++)
			matrix[j] /= val;

		matrix += size_x2;
	}
}

void Uxinv(double *umatrix, double *mul_umatrix_inv, double *umatrix_aux, int iter,int num_bands){
	int j, k, div, aux1, aux2, modulus;
	double valor;

	for(j=0;j<num_bands*iter;j+=1){
		valor=0.0;
		div=(int)(j/iter);
		aux1=j/iter;
		aux2=iter*aux1;
		modulus=j-aux2;
		for(k=0;k<iter;k+=1)
			valor+=umatrix[div+num_bands*k]*mul_umatrix_inv[modulus+k*iter];
		umatrix_aux[j]=valor;			
	}
}

void AnsxUt(double *umatrix_aux, double *umatrix, double *proymatrix, int iter, int num_bands){
	int j, k, div, aux1, aux2, modulus;
	double valor;

	for(j = 0; j < num_bands*num_bands; j++){
		valor=0.0;
		div=(int)(j/num_bands);
	 	aux1=j/num_bands;
		aux2=num_bands*aux1;
		modulus=j-aux2;
		for(k=0;k<iter;k+=1)
			valor+=umatrix_aux[k+iter*div]*umatrix[modulus+k*num_bands];
		proymatrix[j]=valor;			
	}
}

void SustractIdentity(double *proymatrix, int num_bands){
	int j, aux1, aux2, modulus;

	for(j = 0; j < num_bands*num_bands; j++){
		aux1=j/(num_bands+1);
		aux2=(num_bands+1)*aux1;
		modulus=j-aux2;

		if (modulus == 0)
			proymatrix[j]=1-proymatrix[j];
		else
			proymatrix[j]=0-proymatrix[j];
	}
}

/// Funcion que realiza el test de newman person con theta y Rsmall y devuelve r_to_inf en el algoritmo gene.
double GENE_NP_test(double* theta, int Nmax, int i, double* M, double* y, double* invRsmall){

/* OJO AL PARCHE EN VEZ DE CALCULAR INV(VAR_B) todo el rato, lo que hago es calcular la inversa de Rsmall de antemano en invRsmall
  de tal forma que calculo la inversa  de var_b como invRsmall / (1+zeta) en vez de inverasa((1+zeta) * Rsmall) 
*/

	double b_vector[Nmax];

	double zeta;

	double var_b[(Nmax-1)* (Nmax-1)];

	double r;
	double zero_to_r;
	double r_to_inf;


	memcpy(b_vector,y,(Nmax)*sizeof(double));

	double alpha, beta;

	int uno = 1;
	int  k = 0;
	int j;

	// theta = a_k-A_k_1*theta_opt;

	alpha = -1,
	beta = 1;
	dgemm_("N", "N", &Nmax, &uno, &i, &alpha, M, &Nmax, theta, &i, &beta, b_vector, &Nmax);



	// zeta = theta'*theta
	zeta = 0;
	for (j = 0; j < i;j++){
		zeta = zeta + theta[j]*theta[j];
	}


	// var_b = (1+zeta) * Rsmall
	// y calcular la inversa de var_b
	// OJO en vez de hacer inv ( (1+zeta)*Rsmall ) lo que se hace es inv(Rsmall) / (1+zeta)
	// así podemos tener la inversa de Rsmall precalculada y ahorrarnos hacer una inversa  en cada iteracion

	for (j = 0; j < (Nmax-1)*(Nmax-1); j++){
		var_b[j] = invRsmall[j] / (1+zeta);
	}


	///r=(b_vector'*inv(var_b)*b_vector);
	r = 0;
	double dot;
	for (k = 0 ; k < Nmax-1;k++){
		dot = 0;
		for (j = 0; j < Nmax-1;j++){
			dot = dot + var_b[k*(Nmax-1) + j] * b_vector[j];
		}
		r = r + dot*b_vector[k];
	}

	r_to_inf = 1- gsl_sf_gamma_inc_Q((double)r/2, (double)(Nmax-1)/(double)2);

	//printf("it %d, r = %.20f\n",i,r_to_inf);
	//fflush(stdout);

	return r_to_inf;

}





