#include "lsu.h"





int writeResult(boost::numeric::ublas::matrix<double> image, const char* filename, int lines, int samples, int bands, int dataType, char* interleave){

	short int *imageSI;
	float *imageF;
	double *imageD;

	int i,j,k,op;

	if(interleave == NULL)
		op = 0;
	else
	{
		if(strcmp(interleave, "bsq") == 0) op = 0;
		if(strcmp(interleave, "bip") == 0) op = 1;
		if(strcmp(interleave, "bil") == 0) op = 2;
	}

	if(dataType == 2)
	{
		imageSI = (short int*)malloc(lines*samples*bands*sizeof(short int));

        	switch(op)
        	{
			case 0:
				for(i=0; i<lines*samples; i++)
					for(j=0; j<bands; j++)
						imageSI[i+j*samples*lines] = (short int)image(i,j);
				break;

			/*case 1:
				for(i=0; i<bands; i++)//puede que no funcione
					for(j=0; j<lines*samples; j++)
						imageSI[j*bands + i] = (short int)image[i*lines*samples + j];
				break;

			case 2:
				for(i=0; i<lines; i++)//puede que no funcione
					for(j=0; j<bands; j++)
						for(k=0; k<samples; k++)
							imageSI[i*bands*samples + (j*samples + k)] = (short int)image[j*lines*samples + (i*samples + k)];
				break;*/
        	}
		FILE *fp;
    		if ((fp=fopen(filename,"wb"))!=NULL){
        		fseek(fp,0L,SEEK_SET);

	        	fwrite(imageSI,1,(lines*samples*bands * sizeof(short int)),fp);
		}
		free(imageSI);
		printf("File with endmembers saved at: %s\n",filename);
		return 0;

	}

	if(dataType == 4)
	{
		imageF = (float*)malloc(lines*samples*bands*sizeof(float));
        	switch(op)
        	{
			case 0:
				for(i=0; i<lines*samples; i++)
					for(j=0; j<bands; j++)
						imageF[i+j*samples*lines] = (short int)image(i,j);
				break;

			/*case 1:
				for(i=0; i<bands; i++)//puede que no funcione
					for(j=0; j<lines*samples; j++)
						imageF[j*bands + i] = (float)image[i*lines*samples + j];
				break;

			case 2:
				for(i=0; i<lines; i++)//puede que no funcione
					for(j=0; j<bands; j++)
						for(k=0; k<samples; k++)
							imageF[i*bands*samples + (j*samples + k)] = (float)image[j*lines*samples + (i*samples + k)];
				break;*/
        	}
		FILE *fp;
    		if ((fp=fopen(filename,"wb"))!=NULL){
        		fseek(fp,0L,SEEK_SET);

	        	fwrite(imageF,1,(lines*samples*bands * sizeof(float)),fp);
		}
		free(imageF);
		printf("File with endmembers saved at: %s\n",filename);
		return 0;
	}

	if(dataType == 5)
	{
		imageD = (double*)malloc(lines*samples*bands*sizeof(double));
        	switch(op)
        	{
			case 0:
				for(i=0; i<lines*samples; i++)
					for(j=0; j<bands; j++)
						imageD[i+j*samples*lines] = (short int)image(i,j);
				break;

			/*case 1:
				for(i=0; i<bands; i++)//puede que no funcione
					for(j=0; j<lines*samples; j++)
						imageD[j*bands + i] = image[i*lines*samples + j];
				break;

			case 2:
				for(i=0; i<lines; i++)//puede que no funcione
					for(j=0; j<bands; j++)
						for(k=0; k<samples; k++)
							imageD[i*bands*samples + (j*samples + k)] = image[j*lines*samples + (i*samples + k)];
				break;*/
        	}

		FILE *fp;
    		if ((fp=fopen(filename,"wb"))!=NULL){
        		fseek(fp,0L,SEEK_SET);

	        	fwrite(imageD,1,(lines*samples*bands * sizeof(double)),fp);
		}
		free(imageD);
		printf("File with endmembers saved at: %s\n",filename);
		return 0; 
	}



    return -3;
}



void lsu_gpu_v(float *imagen, float *endmembers, int DeviceSelected, int bandas, int targets, int lines, int samples, char *filename){
	
	int i, j, one = 1;
	int sampleLines = lines * samples;
	boost::numeric::ublas::matrix<double> imagen_Host(sampleLines, bandas);
	boost::numeric::ublas::matrix<double> endmember_Host(targets, bandas);
	for(i = 0; i < sampleLines; i++)
		for(j=0; j<bandas; j++)
			imagen_Host(i,j) = imagen[i + lines*samples*j];

	for(i = 0; i < targets; i++)
		for(j=0; j<bandas; j++)
			endmember_Host(i,j) = endmembers[i + targets*j];

	double start = get_time();


	viennacl::ocl::context();
	if(DeviceSelected == 0){
		 viennacl::ocl::set_context_device_type(0, viennacl::ocl::cpu_tag());
	}else if(DeviceSelected == 1){
		viennacl::ocl::set_context_device_type(0, viennacl::ocl::gpu_tag());
	}else if(DeviceSelected == 2){
		viennacl::ocl::set_context_device_type(0, viennacl::ocl::accelerator_tag());
	}
	else{
		printf("Can't find platform selected. Terminating...");
		exit(-1);
	}
	

	std::cout << viennacl::ocl::current_device().info() << std::endl;
	viennacl::context ctx(viennacl::ocl::get_context(0));

    	std::vector<viennacl::ocl::device> devices = viennacl::ocl::current_context().devices();
    	viennacl::ocl::current_context().switch_device(devices[0]);

  	viennacl::tools::timer timer;
  	timer.start();


    	//Host
  	boost::numeric::ublas::matrix<double> EtE_Host_aux(targets, targets);
  	boost::numeric::ublas::matrix<double> EtE_Host(targets, targets);
	boost::numeric::ublas::permutation_matrix<size_t> permutation_Host(targets);
	boost::numeric::ublas::vector<double> AUX_Host(targets);
	boost::numeric::ublas::vector<double> AUX2_Host(targets);
	boost::numeric::ublas::matrix<double> I_Host(targets, targets);
	boost::numeric::ublas::vector<double> PIXEL_Host(bandas);
	double Y_Host;

    	//Device
	viennacl::matrix<double, viennacl::column_major> endmember_Device(targets, bandas, ctx);
    	viennacl::matrix<double> EtE_Device(targets, targets);
	viennacl::vector<double> ONE_Device = viennacl::scalar_vector<double>(targets, 1);
    	viennacl::vector<double> AUX_Device(targets);
   	viennacl::vector<double> AUX2_Device(targets);
	viennacl::scalar<double> Y_Device = double(1.0);
	viennacl::matrix<double> I_Device(targets, targets);
	viennacl::matrix<double> A_Device(targets, targets);
	viennacl::matrix<double> B_Device(bandas, targets);
	viennacl::vector<double> PIXEL_Device(bandas);



	printf("---\n");
	//std::cout << endmember_Host << std::endl;

	double t0 = timer.get();
	viennacl::copy(endmember_Host, endmember_Device);
	double t1 = timer.get();
	printf("Tiempo EtE(transfer->device): %f \n", t1-t0);

	t0 = timer.get();
	EtE_Device = viennacl::linalg::prod(endmember_Device, viennacl::trans(endmember_Device));//esta multiplicacion era column major, es row major(posible error)
	t1 = timer.get();
	printf("Tiempo EtE(prod): %f \n", t1-t0);

	t0 = timer.get();
	viennacl::copy(EtE_Device, EtE_Host_aux);
	t1 = timer.get();
	printf("Tiempo EtE(transfer->host): %f \n", t1-t0);


	t0 = timer.get();
	boost::numeric::ublas::lu_factorize(EtE_Host_aux, permutation_Host);
	boost::numeric::ublas::lu_substitute(EtE_Host_aux, permutation_Host, EtE_Host);
	t1 = timer.get();
	printf("Tiempo EtE(lu): %f \n", t1-t0);


	t0 = timer.get();
	viennacl::copy(EtE_Host, EtE_Device);
	t1 = timer.get();
	printf("Tiempo EtE(transfer->device): %f \n-------\n", t1-t0);


	//esta prueba es para hacerlo todo en serie y mandarlo a device ya calculado
/* 	t0 = timer.get();//esta solo es una prueba de tiempos, por si fuera mejor hacerlo todo con ublas -> NO LO ES!!
	EtE_Host_aux = boost::numeric::ublas::prod(endmember_Host, boost::numeric::ublas::trans(endmember_Host));
	t1 = timer.get();

	std::cout << "EtE_Host_aux: " << EtE_Host_aux << std::endl;

	boost::numeric::ublas::lu_factorize(EtE_Host_aux, pm);
	boost::numeric::ublas::lu_substitute(EtE_Host_aux, pm, EtE_Host);
	t2 = timer.get();

	viennacl::copy(EtE_Host, EtE_Device);
	t3 = timer.get();
	std::cout << "EtE_Host: " << EtE_Host << std::endl;

	printf("Tiempo 2: %f \n", t3-t0);//Tiempo 2: 0.008230
*/
	
	//EtE_Device = viennacl::scalar_matrix<double>(targets, targets, 2);
	t0 = timer.get();
	AUX_Device = viennacl::linalg::prod(EtE_Device, ONE_Device);//Et_E
	t1 = timer.get();
	printf("Tiempo Aux_device: %f \n", t1-t0);

	t0 = timer.get();
	Y_Device = viennacl::linalg::sum(AUX_Device);
	t1 = timer.get();
	printf("Tiempo Y_device(reduction): %f \n", t1-t0);
	Y_Device = 1 / Y_Device;
	Y_Host = Y_Device; 
	//printf("valor Y_Host: %f\n",Y_Host);
	t0 = timer.get();
	AUX_Device = viennacl::scalar_vector<double>(targets, Y_Device);
	t1 = timer.get();
	printf("Tiempo AUX_Device(init): %f \n-------\n", t1-t0);
	
	t0 = timer.get();
	AUX2_Device = viennacl::linalg::prod(EtE_Device, AUX_Device);
	t1 = timer.get();
	printf("Tiempo AUX_Device(prod): %f \n", t1-t0);
	

	t0 = timer.get();
	viennacl::copy(AUX2_Device, AUX2_Host);//me interesa mas traermelo al completo, modificarlo y volverlo a llevar
	t1 = timer.get();
	printf("Tiempo copy Device->Host(aux2): %f \n", t1-t0);
	
	t0 = timer.get();
	for(i=0; i<targets; i++)
		for(j=0; j<targets; j++)
			if(i == j)
				I_Host(i,j) = 1 - AUX2_Host(j);
			else
				I_Host(i,j) = -AUX2_Host(j);
	t1 = timer.get();
	printf("Tiempo host work(I): %f \n", t1-t0);

	t0 = timer.get();
	viennacl::copy(I_Host, I_Device);
	t1 = timer.get();
	printf("Tiempo host->device(I): %f \n", t1-t0);


	t0 = timer.get();
	A_Device = viennacl::linalg::prod(I_Device, EtE_Device);
	t1 = timer.get();
	printf("Tiempo A_Device: %f \n", t1-t0);
	
	//A_Device = viennacl::identity_matrix<double>(targets);//---

	t0 = timer.get();
	B_Device = viennacl::linalg::prod(viennacl::trans(endmember_Device),A_Device);//ojo que esta alreves 188*19
	t1 = timer.get();
	printf("Tiempo B_Device: %f \n", t1-t0);//hacer la transpuesta tarda mucho, alternativas 

	t0 = timer.get();
	AUX_Device = viennacl::linalg::prod(EtE_Device, ONE_Device);
	t1 = timer.get();
	printf("Tiempo AUX_Device(prod): %f \n", t1-t0);

	t0 = timer.get();
	AUX_Device = Y_Device * AUX_Device;
	t1 = timer.get();
	printf("Tiempo Aux_Device (AUX_Device * Y_Device): %f \n", t1-t0);




	t0 = timer.get();
	viennacl::copy(AUX_Device, AUX_Host);
	t1 = timer.get();
	printf("Tiempo Aux_Device(transfer): %f \n", t1-t0);
	
	t0 = timer.get(); 
	for(i = 0; i < lines*samples; i++){
		//PIXEL_Device = viennacl::row(imagen_Device,i);//es muy lenta esta operacion
		for(j = 0; j < bandas; j++) PIXEL_Host(j) = imagen_Host(i,j);
		viennacl::copy(PIXEL_Host, PIXEL_Device);

		AUX2_Device = viennacl::linalg::prod(viennacl::trans(B_Device), PIXEL_Device);//B * Pixel

		AUX2_Device = AUX2_Device + AUX_Device;
	
		viennacl::copy(AUX2_Device, AUX2_Host);

		for(j = 0; j < targets; j++) imagen_Host(i,j) = AUX2_Host(j) + AUX_Host(j);//opcion mas directa como la de extraer row?
		
	}
	t1 = timer.get();
	printf("Tiempo image modifications: %f \n", t1-t0);

	

	viennacl::backend::finish();




	//END CLOCK*****************************************
	double end = get_time();
	printf("Iterative SCLSU: %f segundos\n", (end - start) );
	fflush(stdout);
	//**************************************************

	char results_filename[MAXCAD];
	strcpy(results_filename, filename);
	strcat(results_filename, "Results.hdr");
	writeHeader(results_filename, samples, lines, targets);


	strcpy(results_filename, filename);
	strcat(results_filename, "Results.bsq");
	writeResult(imagen_Host, results_filename, lines, samples, targets, 5, NULL);

	//FREE MEMORY***************************************

	/*free(wavelength);
	free(interleaveE);
	free(interleave);
	free(waveUnit);*/

	/*free(EtE_Host_aux);
	free(EtE_Host);
	free(permutation_Host);
	free(AUX_Host);
	free(AUX2_Host);
	free(I_Host);
	free(PIXEL_Host);
	free(endmember_Host);
	free(imagen_Host);

	free(endmember_Device);
	free(EtE_Device);
	free(ONE_Device);
	free(AUX_Device);
	free(AUX2_Device);
	free(Y_Device);
	free(I_Device);
	free(A_Device);
	free(B_Device);
	free(PIXEL_Device);*/


}

void lsu_gpu_m(float *image, float *endmembers, int DeviceSelected, int bandas, int targets, int lines, int samples, char *filename/*, cl_device_id deviceID*/){


	int i, j, one = 1;
	int sampleLines = lines * samples;

	double *endmember_Host, *image_Host;//se puede mejorar(cambiarlo todo a double?)
	MALLOC_HOST(endmember_Host, double, targets*bandas)
	MALLOC_HOST(image_Host, double, lines*samples*bandas)
	for(i = 0; i < sampleLines*bandas; i++)
		image_Host[i] = (double) image[i];

	for(i = 0; i < targets*bandas; i++)
		endmember_Host[i] = endmembers[i];

	cl_int status;
	unsigned int ok = 0;

    	// determine number of platforms
    	cl_uint numPlatforms;
    	status = clGetPlatformIDs(0, NULL, &numPlatforms); //num_platforms returns the number of OpenCL platforms available
    	exitOnFail2(status, "number of platforms");
	if (CL_SUCCESS == status){
		printf("\nNumber of OpenCL platforms: %d\n", numPlatforms);
		printf("\n-------------------------\n");
	}

	// get platform IDs
  	cl_platform_id platformIDs[numPlatforms];
    	status = clGetPlatformIDs(numPlatforms, platformIDs, NULL); //platformsIDs returns a list of OpenCL platforms found. 
    	exitOnFail2(status, "get platform IDs");

	cl_uint numDevices;
	//cl_platform_id platformID;
        cl_device_id deviceID;
	
	//deviceSelected-> 0:CPU, 1:GPU, 2:ACCELERATOR
	int isCPU = 0, isGPU = 0, isACCEL=0;
	if(DeviceSelected == 0){
		isCPU=1;
	}
	else if(DeviceSelected == 1){
		isGPU=1;
	}
	else if(DeviceSelected == 2){
		isACCEL=1;
	}
	else{	
		printf("Selected device not found. Program will terminate\n");
		exit(-1);
	}
	// iterate over platforms
	for (i = 0; i < numPlatforms; i++){
		// determine number of devices for a platform
		status = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		exitOnFail2(status, "number of devices");
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


	magma_queue_t  queue;
	magma_int_t err;
	magma_init();//falla ponerle esta funcion??
	magma_print_environment();	
	

	err = magma_queue_create( deviceID, &queue );
	if ( err != 0 ) {
	  fprintf( stderr, "magma_queue_create failed: %d\n", (int) err );
	  exit(-1);
	}

	real_Double_t dev_time, total_time;
	double alpha = 1.0;
	double beta  = 0.0;
	size_t size = 0;
	int info;
	int lwork = targets;//redundante ¿?
	double Y = 0;
	//DEVICE
	magmaDouble_ptr endmember_Device, image_Device, EtE_Device, work_Device, one_Device, aux_Device, aux2_Device, I_Device, A_Device, B_Device, pixel_Device;
	MALLOC_DEVICE(endmember_Device, double, targets*bandas)
	MALLOC_DEVICE(image_Device, double, lines*samples*bandas)
	MALLOC_DEVICE(EtE_Device, double, targets*targets)
	MALLOC_DEVICE(work_Device, double, lwork)//probar memoria pinned <- not supported
	MALLOC_DEVICE(one_Device, double, targets)
	MALLOC_DEVICE(aux_Device, double, targets)
	MALLOC_DEVICE(aux2_Device, double, targets)
	MALLOC_DEVICE(I_Device, double, targets*targets)
	MALLOC_DEVICE(A_Device, double, targets*targets)
	MALLOC_DEVICE(B_Device, double, targets*bandas)
	MALLOC_DEVICE(pixel_Device, double, bandas)


	//HOST
	double *EtE_Host, *one_Host, *aux_Host, *aux2_Host, *I_Host, *pixel_Host, *abundancias_Host;
	magma_int_t *ipiv_Host;
	MALLOC_HOST(EtE_Host, double, targets*targets)
	MALLOC_HOST(ipiv_Host, magma_int_t, targets)//pivote de la inversion
	MALLOC_HOST(one_Host, double, targets)
	MALLOC_HOST(aux_Host, double, targets)
	MALLOC_HOST(aux2_Host, double, targets)
	MALLOC_HOST(I_Host, double, targets*targets)
	MALLOC_HOST(pixel_Host, double, bandas)
	MALLOC_HOST(abundancias_Host, double, lines*samples*targets)


	


 	//magma_roundup -> ((m + 31)/32)*32//podria mejorar ligeramente, segun los testing

	total_time = magma_sync_wtime(queue);
	magma_dsetmatrix(targets, bandas, endmember_Host, targets, endmember_Device, size, targets, queue);
	dev_time = magma_sync_wtime(queue) - total_time;
	printf("EtE_transfer host -> device: %f\n",dev_time);

	dev_time = magma_sync_wtime(queue);
	magma_dgemm(MagmaNoTrans, MagmaTrans, targets, targets, bandas, alpha, endmember_Device, size, targets, endmember_Device, size, targets, beta,  EtE_Device, size, targets, queue);
	dev_time = magma_sync_wtime(queue) - dev_time;
	printf("EtE_sgemm: %f\n",dev_time);


	dev_time = magma_sync_wtime(queue);
	magma_dgetmatrix(targets, targets, EtE_Device, size, targets, EtE_Host, targets, queue);
	dev_time = magma_sync_wtime(queue) - dev_time;
	printf("EtE_transfer host -> device: %f\n",dev_time);
	//for(int m = 0; m < targets*targets; m++) printf("%f ",EtE_Host[m]); printf("\n");

/*
	dev_time = magma_sync_wtime(queue);
	magma_dgetrf_gpu( targets, targets, EtE_Device, size, targets, ipiv_Host, queue, &info);
	//(magma_int_t, magmaDouble_ptr, size_t, magma_int_t, magma_int_t*, magmaDouble_ptr, size_t, magma_int_t, _cl_command_queue**, magma_int_t*)	
	magma_dgetri_gpu(targets, EtE_Device, size, targets, ipiv_Host, work_Device, size, lwork, &queue, &info);//parametro 6 tiene ilegal value ¿?¿?¿?¿?
	dev_time = magma_sync_wtime(queue) - dev_time;
	printf("EtE_sgemm: %f\n",dev_time);
*/

	double *work = (double*)malloc(lwork*sizeof(double));
	//dgetrf_(&targets, &targets, EtE_Host, &targets, ipiv, &info);
	lapackf77_dgetrf( &targets, &targets, EtE_Host, &targets, ipiv_Host, &info );
	//dgetri_(&targets, EtE_Host, &targets, ipiv_Host, work, &lwork, &info);
	lapackf77_dgetri( &targets, EtE_Host, &targets, ipiv_Host, work, &lwork, &info );
	magma_dsetmatrix(targets, targets, EtE_Host, targets, EtE_Device, size, targets, queue);


	dev_time = magma_sync_wtime(queue);
	for(int m = 0; m < targets; m++) one_Host[m] = 1.0;
	magma_dsetmatrix(targets, one, one_Host, targets, one_Device, size, targets, queue);
	magma_dgemm(MagmaNoTrans, MagmaNoTrans, one, targets, targets, alpha, one_Device, size, one, EtE_Device, size, targets, beta, aux_Device, size, one, queue);
	magma_dgetmatrix(targets, one, aux_Device, size, targets, aux_Host, targets, queue);
	dev_time = magma_sync_wtime(queue) - dev_time;
	printf("Aux_dgemm: %f\n",dev_time);

	
	dev_time = magma_sync_wtime(queue);	
	for(int m = 0; m < targets; m++) Y += aux_Host[m];
	Y = 1 / Y;
	for(int m = 0; m < targets; m++) aux_Host[m] = Y;
	magma_dsetmatrix(targets, one, aux_Host, targets, aux_Device, size, targets, queue);
	dev_time = magma_sync_wtime(queue) - dev_time;
	printf("aux_reduce(cpu): %f\n",dev_time);


	dev_time = magma_sync_wtime(queue);
	magma_dgemm(MagmaNoTrans, MagmaNoTrans, one, targets, targets, alpha, aux_Device, size, one, EtE_Device, size, targets, beta, aux2_Device, size, one, queue);
	magma_dgetmatrix(targets, one, aux2_Device, size, targets, aux2_Host, targets, queue);


	for(int m = 0; m < targets; m++)
		for(int n = 0; n < targets; n++)
			if(m == n) I_Host[m*targets+n] = 1 - aux2_Host[n];
			else I_Host[m*targets+n] = -aux2_Host[n];
	
	magma_dsetmatrix(targets, targets, I_Host, targets, I_Device, size, targets, queue);
	dev_time = magma_sync_wtime(queue) - dev_time;
	printf("aux2_dgemm / I (init): %f\n",dev_time);


	dev_time = magma_sync_wtime(queue);
	magma_dgemm(MagmaNoTrans, MagmaNoTrans, targets, targets, targets, alpha, I_Device, size, targets, EtE_Device, size, targets, beta, A_Device, size, targets, queue);
	dev_time = magma_sync_wtime(queue) - dev_time;
	printf("A_dgemm: %f\n",dev_time);


	dev_time = magma_sync_wtime(queue);
	magma_dgemm(MagmaNoTrans, MagmaNoTrans, targets, bandas, targets, alpha, A_Device, size, targets, endmember_Device, size, targets, beta, B_Device, size, targets, queue);
	dev_time = magma_sync_wtime(queue) - dev_time;
	printf("B_dgemm: %f\n",dev_time);


	dev_time = magma_sync_wtime(queue);
	magma_dgemm(MagmaNoTrans, MagmaNoTrans, targets, one, targets, alpha, EtE_Device, size, targets, one_Device, size, targets, beta, aux_Device, size, targets, queue);
	magma_dgetmatrix(targets, one, aux_Device, size, targets, aux_Host, targets, queue);
	for(int m = 0; m < targets; m++) aux_Host[m] *= Y;
	magma_dsetmatrix(targets, one, aux_Host, targets, aux_Device, size, targets, queue);
	dev_time = magma_sync_wtime(queue) - dev_time;
	printf("Aux_dgemm: %f\n",dev_time);


	dev_time = magma_sync_wtime(queue);
	magma_dgetmatrix(targets, one, aux_Device, size, targets, aux_Host, targets, queue);//si se actualiza ya no lo necesitaria
	for(int m = 0; m < lines*samples; m++){
		for(int n = 0; n < bandas; n++) pixel_Host[n] = image_Host[n*lines*samples+m];
		magma_dsetmatrix(bandas, one, pixel_Host, bandas, pixel_Device, size, bandas, queue);
		magma_dgemm(MagmaNoTrans, MagmaNoTrans, targets, one, bandas, alpha, B_Device, size, targets, pixel_Device, size, bandas, beta, aux2_Device, size, targets, queue);

		//magma_daxpy(targets, alpha, aux_Device, one, aux2_Device, one, queue);//no esta disponible en la version 1.3 <.<!
		magma_dgetmatrix(targets, one, aux2_Device, size, targets, aux2_Host, targets, queue);
		for(int n = 0; n < targets; n++) abundancias_Host[n*lines*samples + m] = aux2_Host[n]+aux_Host[n];
		
		//if(m == 0) for(int n = 0; n < targets; n++) printf("%f ", aux2_Host[n]+aux_Host[n]);
	}
	dev_time = magma_sync_wtime(queue) - dev_time;
	printf("abundancias_solucion: %f\n",dev_time);
	
/*
magma_dsetmatrix(bands, one, pixel_Host, bands, pixel_Device, size, bands, queue);
magma_dgemm(MagmaNoTrans, MagmaNoTrans, targets, one, bands, alpha, B_Device, size, targets, pixel_Device, size, bands, beta, aux2_Device, size, targets, queue);
*/


	total_time = magma_sync_wtime(queue) - total_time;
	printf("Total Time: %f\n",total_time);

	magma_finalize();

	char results_filename[MAXCAD];
	strcpy(results_filename, filename);
	strcat(results_filename, "Results.hdr");
	writeHeader(results_filename, samples, lines, targets);


	strcpy(results_filename, filename);
	strcat(results_filename, "Results.bsq");
	writeResult(abundancias_Host, results_filename, lines, samples, targets);

		

	magma_free_cpu(image_Host);
	magma_free_cpu(endmember_Host);   
	magma_free_cpu(EtE_Host);
	magma_free_cpu(one_Host);
	magma_free_cpu(aux_Host);
	magma_free_cpu(aux2_Host);
	magma_free_cpu(I_Host);
	magma_free_cpu(pixel_Host);
	magma_free_cpu(abundancias_Host);
	magma_free_cpu(ipiv_Host);

	magma_free(endmember_Device);
	magma_free(image_Device);
	magma_free(EtE_Device);
	magma_free(work_Device);
	magma_free(one_Device);
	magma_free(aux_Device);
	magma_free(aux2_Device);
	magma_free(I_Device);
	magma_free(A_Device);
	magma_free(B_Device);
	magma_free(pixel_Device);
}


void exitOnFail2(cl_int status, const char* message){
	if (CL_SUCCESS != status){
		printf("error: %s\n", message);
		printf("error: %d\n", status);
		exit(-1);
	}
}





























