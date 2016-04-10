#include "lsu.h"
//#include "viennacl/linalg/svd.hpp"






void lsu_gpu_v(double *imagen, double *endmembers, int DeviceSelected, int bandas, int targets, int lines, int samples, char *filename){

	int i, j, one = 1;
	int sampleLines = lines * samples;	
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

	double end = get_time();
	printf("Time platform init: %f\n",end-start);

	//std::cout << viennacl::ocl::current_device().info() << std::endl;

  	viennacl::tools::timer timer;
  	timer.start();


	double Y_Host;
	double t0, t1;
	double norm_imagen;
	int lwork = 5*(targets*targets);
	int info = 1;
	double auxk;

    	//Host
	boost::numeric::ublas::matrix<double> imagen_Host(sampleLines, bandas);
	boost::numeric::ublas::matrix<double> endmember_Host(bandas, targets);
	boost::numeric::ublas::matrix<double> MtM_Host(targets, targets);
	boost::numeric::ublas::matrix<double> IFS_Host(targets, targets);
	boost::numeric::ublas::matrix<double> UF_Host(targets, targets);
	boost::numeric::ublas::matrix<double> IF_Host(targets, targets);
	boost::numeric::ublas::matrix<double> IF1_Host(targets, targets);
	boost::numeric::ublas::matrix<double> abundancias_Host(sampleLines, targets);
	double *MtM = (double*)malloc(targets*targets*sizeof(double));
	double *UF = (double*)malloc(targets*targets*sizeof(double));
	double *SF = (double*)malloc(targets*sizeof(double));
	double *V = (double*)malloc(targets*targets*sizeof(double));
	double *work  = (double*)malloc(lwork*sizeof(double));
	double *IFS = (double*)malloc(targets*targets*sizeof(double));
	double *IF = (double*)malloc(targets*targets*sizeof(double));
	double *IF1 = (double*)malloc(targets*targets*sizeof(double));
	double *Aux = (double*)malloc(targets*sizeof(double));
	double *abundancias = (double*)malloc((sampleLines*targets)*sizeof(double));

        //boost::numeric::ublas::matrix<double> endmember_Host_aux(bandas, targets);



	

    	//Device
	viennacl::matrix<double> imagen_Device(sampleLines, bandas);
	viennacl::matrix<double> endmember_Device(bandas, targets);
	viennacl::matrix<double> MtM_Device(targets, targets);
	viennacl::matrix<double> UF_Device(targets, targets);
	viennacl::matrix<double> IFS_Device(targets, targets);
	viennacl::matrix<double> IF_Device(targets, targets);
	viennacl::matrix<double> yy_Device(sampleLines, targets);
	viennacl::matrix<double> IF1_Device(targets, targets);
	viennacl::matrix<double> abundancias_Device(sampleLines, targets);

	//viennacl::matrix<double> endmember_Device_aux(bandas, targets);




	printf("---\n");
	start = get_time();

	t0 = timer.get();
	norm_imagen = avg_X_2(imagen, sampleLines, bandas);
	t1 = timer.get();
	printf("Tiempo norm_y(host): %f \n", t1-t0);


	t0 = timer.get();
	divide_norm(imagen, endmembers, norm_imagen, sampleLines, bandas, targets);
	t1 = timer.get();
	printf("Tiempo divide_norm(host): %f \n", t1-t0);


	t0 = timer.get();
	for(i = 0; i < sampleLines; i++) for(j = 0; j < bandas; j++) imagen_Host(i,j) = imagen[i + j*sampleLines];
	for(i = 0; i < bandas; i++) for(j = 0; j < targets; j++) endmember_Host(i,j) = endmembers[i + j*bandas];
	//for(i = 0; i < bandas; i++) for(j=0; j<targets; j++) endmember_Host_aux(i,j) = endmembers[i + j*bandas];//<----------------
	viennacl::copy(imagen_Host, imagen_Device);
	viennacl::copy(endmember_Host, endmember_Device);
	//viennacl::copy(endmember_Host_aux, endmember_Device_aux);//<----------------------------------
	t1 = timer.get();
	printf("Tiempo transfer Host->Device: %f \n", t1-t0);


	// MtM = M'*M
	t0 = timer.get();
	MtM_Device = viennacl::linalg::prod(viennacl::trans(endmember_Device), endmember_Device);
	//MtM_Device = viennacl::linalg::prod(viennacl::trans(endmember_Device_aux), endmember_Device_aux);//<----------------------------
//for(i = 0; i < targets; i++) for(j=0; j<targets; j++) if(MtM_Device(i,j) != MtM_Device(i,j)) {printf("ERROR: (%d) %f != %f\n",i*targets+j, MtM_Device(i,j),MtM_Device_aux(i,j)); exit(-1);}
	t1 = timer.get();
	printf("Tiempo dgemm(MtM): %f \n", t1-t0);

/* produce resultados incorrectos...
	t0 = timer.get();
	viennacl::linalg::svd(MtM_Device, qr_Device, ql_Device);
	t1 = timer.get();
	printf("Tiempo svd: %f \n", t1-t0);
	sf_Device = viennacl::diag(MtM_Device, 1);
	uf_Device = viennacl::linalg::prod(qr_Device, negativo);
*/

	t0 = timer.get();
	viennacl::copy(MtM_Device, MtM_Host);
	for(i = 0; i < targets; i++) for(j = 0; j < targets; j++) MtM[i*targets+j] = MtM_Host(i,j);
	dgesvd_("A", "N", &targets, &targets, MtM, &targets, SF, UF, &targets, V, &targets, work, &lwork, &info);
	t1 = timer.get();
	printf("Tiempo svd: %f \n", t1-t0);


	t0 = timer.get();
	UFdiag(UF, SF, IFS, targets, 1e-8);//se puede paralelizar pero es muy pequeño => tarda mas en gpu
	t1 = timer.get();
	printf("Tiempo UFdiag: %f \n", t1-t0);

	t0 = timer.get();
	for(i = 0; i < targets; i++) for(j = 0; j < targets; j++){ UF_Host(i,j) = UF[i+j*targets]; IFS_Host(i,j) = IFS[i+j*targets];}
	viennacl::copy(UF_Host, UF_Device);
	viennacl::copy(IFS_Host, IFS_Device);
	t1 = timer.get();
	printf("Tiempo transfer(UF e IFS): %f \n", t1-t0);

	
	t0 = timer.get();
	IF_Device = viennacl::linalg::prod(IFS_Device, viennacl::trans(UF_Device));
	t1 = timer.get();
	printf("Tiempo dgemm(IF): %f \n", t1-t0);	

	//std::cout << IF_Device << std::endl;


	t0 = timer.get();
	viennacl::copy(IF_Device, IF_Host);
	for(i = 0; i < targets; i++) for(j = 0; j < targets; j++) IF[i*targets+j] = IF_Host(i,j);
	IF1_Aux(IF, IF1, Aux, targets);
	t1 = timer.get();
	printf("Tiempo IF1_Aux: %f \n", t1-t0);

	
	t0 = timer.get();
	yy_Device = viennacl::linalg::prod(imagen_Device, endmember_Device);
	t1 = timer.get();
	printf("Tiempo dgemm(yy): %f \n", t1-t0);


/*
boost::numeric::ublas::matrix<double> aa(sampleLines, targets);
viennacl::copy(yy_Device, aa);
double *aaa = (double*)malloc((sampleLines*targets)*sizeof(double));
for(i = 0; i < sampleLines; i++) for(j = 0; j < targets; j++) aaa[i*targets+j] = aa(i,j);

for(i = 0; i < targets*2; i++)  printf("%f ", aaa[i]); printf("\n%f\n",aaa[sampleLines*targets-1]);
*/

	t0 = timer.get();
	for(i = 0; i < targets; i++) for(j = 0; j < targets; j++) IF1_Host(i,j) = IF1[i*targets+j];
	viennacl::copy(IF1_Host, IF1_Device);
	abundancias_Device = viennacl::linalg::prod(yy_Device, IF1_Device);//sale transpuesta
	t1 = timer.get();
	printf("Tiempo dgemm(abundancias): %f \n", t1-t0);


	t0 = timer.get();
	viennacl::copy(abundancias_Device, abundancias_Host);
	for(i = 0; i < sampleLines; i++) for(j = 0; j < targets; j++) abundancias[i+j*sampleLines] = abundancias_Host(i,j);

for(i = 0; i < targets; i++)  printf("%f ", abundancias[i]); printf("\n%f\n",abundancias[sampleLines*targets-1]);







	for (j = 0; j < targets; j++){
		auxk = Aux[j];
		for (i = 0; i < sampleLines; i++){
			abundancias[j*sampleLines+i] = abundancias[j*sampleLines+i] + auxk; 
		}
	}
	t1 = timer.get();
	printf("Tiempo (abundancias + auk): %f \n", t1-t0);	


	viennacl::backend::finish();



	end = get_time();
	printf("Total SCLSU Viennacl Time: %f\n",end-start);


	char results_filename[MAXCAD];
	strcpy(results_filename, filename);
	strcat(results_filename, "Results.hdr");
	writeHeader(results_filename, samples, lines, targets);


	strcpy(results_filename, filename);
	strcat(results_filename, "Results.bsq");
	writeResult(abundancias, results_filename, lines, samples, targets);


	//FREE MEMORY***************************************//


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

void lsu_gpu_m(	double *image_Host,
		double *endmember_Host, 
		int DeviceSelected,//siempre va ser la GPU se podria eliminar 
		int bands, 
		int targets, 
		int lines, 
		int samples, 
		char *filename){

	size_t size = 0;
	double norm_y;
	int ii,k;
	double auxk;
	double alpha = 1, beta = 0;
	int lwork  = 5*(targets*targets);
	magma_int_t info;
	real_Double_t dev_time;
	int linessamples = lines*samples;


	// return code used by OpenCL API
    	cl_int status;
	unsigned int ok = 0, i, j;

    	// determine number of platforms
    	cl_uint numPlatforms;
    	status = clGetPlatformIDs(0, NULL, &numPlatforms); //num_platforms returns the number of OpenCL platforms available
    	exitOnFail2(status, "number of platforms");
	

	// get platform IDs
  	cl_platform_id platformIDs[numPlatforms];
    	status = clGetPlatformIDs(numPlatforms, platformIDs, NULL); //platformsIDs returns a list of OpenCL platforms found. 
    	exitOnFail2(status, "get platform IDs");

	cl_uint numDevices;
	//cl_platform_id platformID;
        cl_device_id deviceID;
	
	//deviceSelected-> 0:CPU, 1:GPU, 2:ACCELERATOR
	int isCPU = 0, isGPU = 1, isACCEL=0;//usaremos la GPU por defecto

	// iterate over platforms
	for (i = 0; i < numPlatforms; i++){
		// determine number of devices for a platform
		status = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		exitOnFail2(status, "number of devices");
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

	//CLMAGMA
	magma_queue_t  queue;
	magma_int_t err;
	magma_init();//falla ponerle esta funcion??
	magma_print_environment();	
	

	err = magma_queue_create( deviceID, &queue );
	if ( err != 0 ) {
	  fprintf( stderr, "magma_queue_create failed: %d\n", (int) err );
	  exit(-1);
	}
	
	dev_time = magma_sync_wtime(queue);
	//device
	magmaDouble_ptr M_d, Y_d, MtM_d, UF_d, SF_d, V_d, work_d, IFS_d, IF_d, IF1_d, yy_d, X_d;
	MALLOC_DEVICE(M_d, double, targets*bands)
	MALLOC_DEVICE(Y_d, double, linessamples*bands)
	MALLOC_DEVICE(MtM_d, double, targets*targets)
	MALLOC_DEVICE(UF_d, double, targets*targets)
	MALLOC_DEVICE(SF_d, double, targets)
	MALLOC_DEVICE(V_d, double, targets*targets)
	MALLOC_DEVICE(work_d, double, lwork)
	MALLOC_DEVICE(IFS_d, double, targets*targets)
	MALLOC_DEVICE(IF_d, double, targets*targets)
	MALLOC_DEVICE(IF1_d, double, targets*targets)
	MALLOC_DEVICE(yy_d, double, targets*linessamples)
	MALLOC_DEVICE(X_d, double, targets*linessamples)


	//host
	double *MtM_h, *SF_h, *UF_h, *V_h, *work_h, *IFS_h, *IF_h, *Aux_h, *IF1_h, *abundancias_h;
	MALLOC_HOST(MtM_h, double, targets*targets)
	MALLOC_HOST(SF_h, double, targets)
	MALLOC_HOST(UF_h, double, targets*targets)
	MALLOC_HOST(V_h, double, targets*targets)
	MALLOC_HOST(work_h, double, lwork)
	MALLOC_HOST(IFS_h, double, targets*targets)
	MALLOC_HOST(IF_h, double, targets*targets)
	MALLOC_HOST(IF1_h, double, targets*targets)
	MALLOC_HOST(Aux_h, double, targets)
	MALLOC_HOST(abundancias_h, double, targets*linessamples)


	dev_time = magma_sync_wtime(queue) - dev_time;
	printf("Reserva e inicializacion de variables: %f\n",dev_time);

	norm_y = avg_X_2(image_Host,linessamples,bands);
	printf("norm_y: %f\n",norm_y);

	divide_norm(image_Host, endmember_Host, norm_y, linessamples, bands, targets);

//--------------------
//printf("M\n");
//for(int z = 0; z < targets*targets; z++) printf("%f ",endmember_Host[z]); printf("\n");
//printf("Y\n");
//for(int z = 0; z < targets*targets; z++) printf("%f ",image_Host[z]); printf("\n");//funciona lol
//-----------------------

	magma_dsetmatrix(targets, bands, endmember_Host, targets, M_d, size, targets, queue);
	magma_dsetmatrix(bands, linessamples, image_Host, bands, Y_d, size, bands, queue);

	dev_time = magma_sync_wtime(queue);
	magma_dgemm(MagmaTrans, MagmaNoTrans, targets, targets, bands, alpha, M_d, size, bands, M_d, size, bands, beta,  MtM_d, size, targets, queue);
	magma_dgetmatrix(targets,targets, MtM_d, size, targets, MtM_h, targets, queue);
	dev_time = magma_sync_wtime(queue) - dev_time;
	printf("dgemm M'*M: %f\n",dev_time);


//--------------------
//printf("dgemm MtM\n");
//for(int z = 0; z < targets*targets; z++) printf("%f ",MtM_h[z]); printf("\n");
//-----------------------

	dev_time = magma_sync_wtime(queue);
	magma_dgesvd(MagmaAllVec, MagmaNoVec, targets, targets, MtM_h, targets, SF_h, UF_h, targets, V_h, targets, work_h, lwork, queue, &info);
	//printf("Info: %d\n", (int)info);
	//magma_dsetmatrix(targets, one, SF_h, targets, SF_d, size, targets, queue);
	magma_dsetmatrix(targets, targets, UF_h, targets, UF_d, size, targets, queue);
	//magma_dsetmatrix(targets, targets, V_h, targets, V_d, size, targets, queue);
	dev_time = magma_sync_wtime(queue) - dev_time;
	printf("dgesvd: %f\n",dev_time);

	UFdiag(UF_h, SF_h, IFS_h, targets, 1e-8);
	magma_dsetmatrix(targets, targets, IFS_h, targets, IFS_d, size, targets, queue);
//--------------------
//printf("dgesv (sf_h) \n");
//for(int z = 0; z < targets; z++) printf("%f ",SF_h[z]); printf("\n");
//printf("dgesv (uf_h)\n");
//for(int z = 0; z < targets*targets; z++) printf("%f ",UF_h[z]); printf("\n");
//printf("dgesv \n");
//for(int z = 0; z < targets*targets; z++) printf("%f ",IFS_h[z]); printf("\n");//funciona
//-----------------------

	dev_time = magma_sync_wtime(queue);
	magma_dgemm(MagmaNoTrans, MagmaTrans, targets, targets, targets, alpha, IFS_d, size, targets, UF_d, size, targets, beta, IF_d, size, targets, queue);
	magma_dgetmatrix(targets, targets, IF_d, size, targets, IF_h, targets, queue);
	dev_time = magma_sync_wtime(queue) - dev_time;
	printf("dgemm IFS*UF': %f\n",dev_time);

//--------------------
//printf("dgemm IF \n");
//for(int z = 0; z < targets*targets; z++) printf("%f ",IF_h[z]); printf("\n");//funciona
//-----------------------

	IF1_Aux(IF_h, IF1_h, Aux_h, targets);



	dev_time = magma_sync_wtime(queue);
	magma_dgemm(MagmaNoTrans, MagmaNoTrans, linessamples, targets, bands, alpha, Y_d, size, linessamples, M_d, size, bands, beta, yy_d, size, linessamples, queue);
	dev_time = magma_sync_wtime(queue) - dev_time;
	printf("dgemm Y*M: %f\n",dev_time);

//--------------------
printf("dgemm Y*M = yy_d\n");
double *yyy;
MALLOC_HOST(yyy, double, targets*linessamples)
magma_dgetmatrix(targets, linessamples, yy_d, size, targets, yyy, targets, queue);
for(int z = 0; z < targets*2; z++) printf("%f ",yyy[z]); printf("\n");
printf("%f ",yyy[targets*linessamples-1]); printf("\n");
//-----------------------

	dev_time = magma_sync_wtime(queue);
	magma_dsetmatrix(targets, targets, IF1_h, targets, IF1_d, size, targets, queue);
	magma_dgemm(MagmaNoTrans, MagmaNoTrans, linessamples, targets, targets, alpha, yy_d, size, linessamples, IF1_d, size, targets, beta, X_d, size, linessamples, queue);
	dev_time = magma_sync_wtime(queue) - dev_time;
	printf("dgemm yy*IF1: %f\n",dev_time);


	magma_dgetmatrix(linessamples,targets, X_d, size, linessamples, abundancias_h, linessamples, queue);

//-------------
//for(int z = 0; z < targets; z++) printf("%f ",abundancias_h[z]);printf("\n%f\n",abundancias_h[linessamples*targets-1]);
//---------------	
	for (k=0; k< targets;k++){
		auxk = Aux_h[k];
		for (ii=0;ii<linessamples;ii++){//se puede paralelizar
			abundancias_h[k*linessamples+ii] = abundancias_h[k*linessamples+ii] + auxk;	
		}
	}

	magma_finalize();

	char results_filename[MAXCAD];
	strcpy(results_filename, filename);
	strcat(results_filename, "Results.hdr");
	writeHeader(results_filename, samples, lines, targets);


	strcpy(results_filename, filename);
	strcat(results_filename, "Results.bsq");
	writeResult(abundancias_h, results_filename, lines, samples, targets);



	magma_free(M_d);
	magma_free(Y_d);
	magma_free(MtM_d);
	magma_free(UF_d);
	magma_free(SF_d);
	magma_free(V_d);
	magma_free(work_d);
	magma_free(IFS_d);
	magma_free(IF_d);
	magma_free(IF1_d);
	magma_free(yy_d);


	magma_free_cpu(MtM_h);
	magma_free_cpu(SF_h);
	magma_free_cpu(UF_h);
	magma_free_cpu(V_h);
	magma_free_cpu(work_h);
	magma_free_cpu(IFS_h);
	magma_free_cpu(IF_h);
	magma_free_cpu(Aux_h);
	magma_free_cpu(IF1_h);
	magma_free_cpu(abundancias_h);


}


void exitOnFail2(cl_int status, const char* message){
	if (CL_SUCCESS != status){
		printf("error: %s\n", message);
		printf("error: %d\n", status);
		exit(-1);
	}
}

double avg_X_2(double *X, int lines_samples, int num_bands){

	int i,j;
    	double mean;
	double value;
	mean=0;

	for(i=0; i < num_bands; i++){
		
        	for(j=0; j < lines_samples; j++){
			value = X[(i*lines_samples)+j];
        		mean += value*value;
		}
    	}

	return(sqrt(mean/(lines_samples*num_bands)));
}

void divide_norm(double *X, double* M, double norm, int lines_samples, int bands, int p){

	int i, j;

	for (i = 0; i < bands;i++){ 
		for (j = 0;j<lines_samples;j++){
			X[i*lines_samples+j] = X[i*lines_samples+j] / norm;
		}
	}


	for (i = 0; i < bands;i++){ 
		for (j = 0;j<p;j++){
			M[i*p+j] = M[i*p+j]/ norm;
		}
	}

}

void UFdiag(double* UF,double* SF,double* IF,int targets,double mu){
	int i,j  = 0;

	for (i = 0; i< targets; i++){
		for (j = 0; j< targets;j++){
			IF[i*targets+j] = UF[i*targets+j]*(1/(SF[i]+mu));
		}
	}
}

void IF1_Aux(double* IF,double* IF1, double* Aux,int targets){

	/// calloc pone a 0s los sumatorios pero pone por byte
	double* sumaFilas = (double*)calloc(targets,sizeof(double));
	double sumaTot;

	int i, j;


	sumaTot = 0;

	for (i = 0; i< targets;i++){
		for (j = 0; j < targets;j++){
			sumaFilas[i] = sumaFilas[i] + IF[i*targets+j];	
			sumaTot = sumaTot + IF[i*targets+j];
		}
	}


	for (i = 0; i< targets;i++){
		for (j = 0; j < targets;j++){
			IF1[i*targets+j] = IF[i*targets+j] - ( sumaFilas[i]* sumaFilas[j] /sumaTot) ;
		}
		Aux[i] = sumaFilas[i] / sumaTot;
	}


	free(sumaFilas);
}




