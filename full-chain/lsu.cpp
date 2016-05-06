#include "lsu.h"
#include "viennacl/coordinate_matrix.hpp"



void lsu_gpu_v(double *imagen, double *endmembers, int DeviceSelected, int bandas, int targets, int lines, int samples, char *filename){

	int i, j, one = 1;
	int sampleLines = lines * samples;	
	double t0Init = get_time();

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

  	viennacl::tools::timer timer;
  	timer.start();


	double Y_Host;
	double t0, t1;
	double tTransfer = 0.0, tExeGPU = 0.0, tExeCPU = 0.0;
	double norm_imagen;
	int lwork = 5*(targets*targets);
	int info = 1;
	double auxk;

    	//Host
	std::vector<std::vector<double> > imagen_vector(sampleLines, std::vector<double>(bandas));
	std::vector<std::vector<double> > endmember_vector(bandas, std::vector<double>(targets));
	std::vector<std::vector<double> > MtM_vector(targets, std::vector<double>(targets));
	std::vector<std::vector<double> > IFS_vector(targets, std::vector<double>(targets));
	std::vector<std::vector<double> > UF_vector(targets, std::vector<double>(targets));
	std::vector<std::vector<double> > IF_vector(targets, std::vector<double>(targets));
	std::vector<std::vector<double> > IF1_vector(targets, std::vector<double>(targets));
	std::vector<std::vector<double> > abundancias_vector(sampleLines, std::vector<double>(targets));
	//std::vector<std::map<unsigned int, double> > prueba(sampleLines);
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

	viennacl::matrix<double> prueba_Device(sampleLines, targets);



	double t1Init = get_time();

	

	printf("-----------------------------------------------------------------------\n");
	printf("                            ViennaCl\n");
	printf("-----------------------------------------------------------------------\n");
	//start = get_time();

	t0 = timer.get();
	norm_imagen = avg_X_2(imagen, sampleLines, bandas);
	t1 = timer.get();
	tExeCPU += t1-t0;
	//printf("Tiempo norm_y(host): %f \n", t1-t0);


	t0 = timer.get();
	divide_norm(imagen, endmembers, norm_imagen, sampleLines, bandas, targets);
	t1 = timer.get();
	tExeCPU += t1-t0;
	//printf("Tiempo divide_norm(host): %f \n", t1-t0);


	
	for (i = 0; i < sampleLines; i++){
		for (j = 0; j < bandas; j++){
			imagen_vector[i][j] = imagen[i + j*sampleLines];
		}
	}
	for (i = 0; i < bandas; i++){
		for (j = 0; j < targets; j++){
			endmember_vector[i][j] = endmembers[i + j*bandas];
		}
	}
	//he decicido tomar el tiempo de transferencia de la imagen como si viniera en formato vector<vector<double> > puesto que podria leer la imagen en vez de 
	//double* directamente en formato vector<vector<double> > solo tendria que sobrecargar todas las funciones que se usan en SGA...
	//asi puedo comparar los tiempos de transferencia con los de Magma
	t0 = timer.get();
	viennacl::copy(imagen_vector, imagen_Device);
	viennacl::copy(endmember_vector, endmember_Device);
	t1 = timer.get();
	tTransfer += t1-t0;
	printf("Tiempo transfer Host->Device(imagen): %f \n", t1-t0);



	// MtM = M'*M
	t0 = timer.get();
	MtM_Device = viennacl::linalg::prod(viennacl::trans(endmember_Device), endmember_Device);
	t1 = timer.get();
	tExeGPU += t1-t0;
	//printf("Tiempo dgemm(MtM): %f \n", t1-t0);

/* produce resultados incorrectos...
	t0 = timer.get();
	viennacl::linalg::svd(MtM_Device, qr_Device, ql_Device);
	t1 = timer.get();
	printf("Tiempo svd: %f \n", t1-t0);
	sf_Device = viennacl::diag(MtM_Device, 1);
	uf_Device = viennacl::linalg::prod(qr_Device, negativo);
*/

	t0 = timer.get();
	viennacl::copy(MtM_Device, MtM_vector);
	for(i = 0; i < targets; i++) for(j = 0; j < targets; j++) MtM[i*targets+j] = MtM_vector[i][j];
	t1 = timer.get();
	tTransfer += t1-t0;

	t0 = timer.get();
	dgesvd_("A", "N", &targets, &targets, MtM, &targets, SF, UF, &targets, V, &targets, work, &lwork, &info);
	t1 = timer.get();
	tExeCPU += t1-t0;


	t0 = timer.get();
	UFdiag(UF, SF, IFS, targets, 1e-8);//se puede paralelizar pero es muy pequeño => tarda mas en gpu
	t1 = timer.get();
	tExeCPU += t1-t0;
	//printf("Tiempo UFdiag: %f \n", t1-t0);

	t0 = timer.get();
	for(i = 0; i < targets; i++) for(j = 0; j < targets; j++){ UF_vector[i][j] = UF[i+j*targets]; IFS_vector[i][j] = IFS[i+j*targets];}
	viennacl::copy(UF_vector, UF_Device);
	viennacl::copy(IFS_vector, IFS_Device);
	t1 = timer.get();
	tTransfer += t1-t0;
	//printf("Tiempo transfer(UF e IFS): %f \n", t1-t0);

	
	t0 = timer.get();
	IF_Device = viennacl::linalg::prod(IFS_Device, viennacl::trans(UF_Device));
	t1 = timer.get();
	tExeGPU += t1-t0;
	//printf("Tiempo dgemm(IF): %f \n", t1-t0);	



	t0 = timer.get();
	viennacl::copy(IF_Device, IF_vector);
	for(i = 0; i < targets; i++) for(j = 0; j < targets; j++) IF[i*targets+j] = IF_vector[i][j];
	t1 = timer.get();
	tTransfer += t1-t0;

	t0 = timer.get();
	IF1_Aux(IF, IF1, Aux, targets);
	t1 = timer.get();
	tExeCPU += t1-t0;
	//printf("Tiempo IF1_Aux: %f \n", t1-t0);

	
	t0 = timer.get();
	yy_Device = viennacl::linalg::prod(imagen_Device, endmember_Device);
	t1 = timer.get();
	tExeGPU += t1-t0;
	//printf("Tiempo dgemm(yy): %f \n", t1-t0);



	t0 = timer.get();
	for(i = 0; i < targets; i++) for(j = 0; j < targets; j++) IF1_vector[i][j] = IF1[i*targets+j];
	viennacl::copy(IF1_vector, IF1_Device);
	t1 = timer.get();
	tTransfer += t1-t0;

	t0 = timer.get();
	abundancias_Device = viennacl::linalg::prod(yy_Device, IF1_Device);
	t1 = timer.get();
	tExeGPU += t1-t0;
	//printf("Tiempo dgemm(abundancias): %f \n", t1-t0);


	t0 = timer.get();
	viennacl::copy(abundancias_Device, abundancias_vector);
	for(i = 0; i < sampleLines; i++) for(j = 0; j < targets; j++) abundancias[i+j*sampleLines] = abundancias_vector[i][j];
	t1 = timer.get();
	tTransfer += t1-t0;	
	//for(i = 0; i < targets; i++)  printf("%f ", abundancias_vector[i][0]); printf("\n%f\n",abundancias_vector[sampleLines-1][targets-1]);//<- borrar

	t0 = timer.get();
	for (j = 0; j < targets; j++){
		auxk = Aux[j];
		for (i = 0; i < sampleLines; i++){
			abundancias[j*sampleLines+i] += auxk; 
		}
	}
	t1 = timer.get();
	tExeCPU += t1-t0;
	//printf("Tiempo (abundancias + auk): %f \n", t1-t0);	


	viennacl::backend::finish();


	printf("\nTotal INIT:	\t%.3f (seconds)\n", t1Init-t0Init);
	double tTotal = (t1Init-t0Init)	+ tTransfer + tExeCPU + tExeGPU;
	double tExe =  tTransfer + tExeCPU + tExeGPU;
	printf("\nTotal LSU:	\t%.3f (seconds)\n Transfer:    \t\t%.3f\t(%2.1f%) \n Execution(CPU): \t%.3f\t(%2.1f%)\n Execution(GPU): \t%2.3f\t(%.1f%)\n", tExe, tTransfer, (tTransfer*100)/tExe, tExeCPU, (tExeCPU*100)/tExe, tExeGPU, (tExeGPU*100)/tExe);
	printf("\nTotal:	\t\t%.3f (seconds)\n\n",tTotal);





	char results_filename[MAXCAD];
	strcpy(results_filename, filename);
	strcat(results_filename, "Results.hdr");
	writeHeader(results_filename, samples, lines, targets);


	strcpy(results_filename, filename);
	strcat(results_filename, "Results.bsq");
	writeResult(abundancias, results_filename, samples, lines, targets);


	//FREE MEMORY***************************************//


	/*free(endmember_Device);
	free(EtE_Device);
	free(ONE_Device);
	free(AUX_Device);
	free(AUX2_Device);
	free(Y_Device);
	free(I_Device);
	free(A_Device);
	free(B_Device);
	free(PIXEL_Device);*/

	free(MtM);
	free(UF);
	free(SF);
	free(V);
	free(work);
	free(IFS);
	free(IF);
	free(IF1);
	free(Aux);


	


}

void lsu_gpu_m(	double *image_Host,
		double *endmember_Host, 
		cl_device_id deviceID,
		int bands, 
		int targets, 
		int lines, 
		int samples, 
		char *filename,
		double *abundancias_h){

	size_t size = 0;
	double norm_y;
	int ii,k;
	double auxk;
	double alpha = 1, beta = 0;
	int lwork  = 5*(targets*targets);
	magma_int_t info;
	real_Double_t dev_time;
	int linessamples = lines*samples;

	double tTransfer = 0.0, tExeGPU = 0.0, tExeCPU = 0.0;
	double t0Init = get_time();
	

	if(filename != NULL){
		printf("-----------------------------------------------------------------------\n");
		printf("                            ClMagma\n");
		printf("-----------------------------------------------------------------------\n");
	}
	//CLMAGMA
	magma_queue_t queue;
	magma_int_t err;
	if(filename != NULL){
		magma_init();//falla ponerle esta funcion??
		magma_print_environment();
	}
		
	
	err = magma_queue_create( deviceID, &queue );
	if ( err != 0 ) {
  		fprintf( stderr, "magma_queue_create failed: %d\n", (int) err );
  		exit(-1);
	}
		
		
	

	
	

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
	double *MtM_h, *SF_h, *UF_h, *V_h, *work_h, *IFS_h, *IF_h, *Aux_h, *IF1_h;
	MALLOC_HOST(MtM_h, double, targets*targets)
	MALLOC_HOST(SF_h, double, targets)
	MALLOC_HOST(UF_h, double, targets*targets)
	MALLOC_HOST(V_h, double, targets*targets)
	MALLOC_HOST(work_h, double, lwork)
	MALLOC_HOST(IFS_h, double, targets*targets)
	MALLOC_HOST(IF_h, double, targets*targets)
	MALLOC_HOST(IF1_h, double, targets*targets)
	MALLOC_HOST(Aux_h, double, targets)
	//MALLOC_HOST(abundancias_h, double, targets*linessamples)

	double t1Init = get_time();


	dev_time = magma_sync_wtime(queue);
	norm_y = avg_X_2(image_Host,linessamples,bands);
	divide_norm(image_Host, endmember_Host, norm_y, linessamples, bands, targets);
	tExeCPU += magma_sync_wtime(queue) - dev_time;


//--------------------
//printf("M\n");
//for(int z = 0; z < targets*targets; z++) printf("%f ",endmember_Host[z]); printf("\n");
//printf("Y\n");
//for(int z = 0; z < targets*targets; z++) printf("%f ",image_Host[z]); printf("\n");//funciona lol
//-----------------------

	dev_time = magma_sync_wtime(queue);
	magma_dsetmatrix(targets, bands, endmember_Host, targets, M_d, size, targets, queue);
	magma_dsetmatrix(bands, linessamples, image_Host, bands, Y_d, size, bands, queue);
	tTransfer += magma_sync_wtime(queue) - dev_time;
	//printf("transfer image: %f\n",dev_time);


	dev_time = magma_sync_wtime(queue);
	magma_dgemm(MagmaTrans, MagmaNoTrans, targets, targets, bands, alpha, M_d, size, bands, M_d, size, bands, beta,  MtM_d, size, targets, queue);
	tExeGPU += magma_sync_wtime(queue) - dev_time;


	dev_time = magma_sync_wtime(queue);
	magma_dgetmatrix(targets,targets, MtM_d, size, targets, MtM_h, targets, queue);
	tTransfer += magma_sync_wtime(queue) - dev_time;
	//printf("dgemm M'*M: %f\n",dev_time);


//--------------------
//printf("dgemm MtM\n");
//for(int z = 0; z < targets*targets; z++) printf("%f ",MtM_h[z]); printf("\n");
//-----------------------

	dev_time = magma_sync_wtime(queue);
	dgesvd_("A", "N", &targets, &targets, MtM_h, &targets, SF_h, UF_h, &targets, V_h, &targets, work_h, &lwork, &info);
	//magma_dgesvd(MagmaAllVec, MagmaNoVec, targets, targets, MtM_h, targets, SF_h, UF_h, targets, V_h, targets, work_h, lwork, queue, &info);
	tExeGPU += magma_sync_wtime(queue) - dev_time;


	dev_time = magma_sync_wtime(queue);
	magma_dsetmatrix(targets, targets, UF_h, targets, UF_d, size, targets, queue);
	tTransfer += magma_sync_wtime(queue) - dev_time;
	//printf("dgesvd: %f\n",dev_time);


	dev_time = magma_sync_wtime(queue);
	UFdiag(UF_h, SF_h, IFS_h, targets, 1e-8);
	tExeCPU += magma_sync_wtime(queue) - dev_time;


	dev_time = magma_sync_wtime(queue);
	magma_dsetmatrix(targets, targets, IFS_h, targets, IFS_d, size, targets, queue);
	tTransfer += magma_sync_wtime(queue) - dev_time;

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
	tExeGPU += magma_sync_wtime(queue) - dev_time;


	dev_time = magma_sync_wtime(queue);
	magma_dgetmatrix(targets, targets, IF_d, size, targets, IF_h, targets, queue);
	tTransfer += magma_sync_wtime(queue) - dev_time;
	//printf("dgemm IFS*UF': %f\n",dev_time);

//--------------------
//printf("dgemm IF \n");
//for(int z = 0; z < targets*targets; z++) printf("%f ",IF_h[z]); printf("\n");//funciona
//-----------------------

	dev_time = magma_sync_wtime(queue);
	IF1_Aux(IF_h, IF1_h, Aux_h, targets);
	tExeCPU += magma_sync_wtime(queue) - dev_time;


	dev_time = magma_sync_wtime(queue);
	magma_dgemm(MagmaNoTrans, MagmaNoTrans, linessamples, targets, bands, alpha, Y_d, size, linessamples, M_d, size, bands, beta, yy_d, size, linessamples, queue);
	tExeGPU += magma_sync_wtime(queue) - dev_time;
	//printf("dgemm Y*M: %f\n",dev_time);

//--------------------
//printf("dgemm Y*M = yy_d\n");
//double *yyy;
//MALLOC_HOST(yyy, double, targets*linessamples)
//magma_dgetmatrix(targets, linessamples, yy_d, size, targets, yyy, targets, queue);
//for(int z = 0; z < targets*2; z++) printf("%f ",yyy[z]); printf("\n");
//printf("%f ",yyy[targets*linessamples-1]); printf("\n");
//-----------------------

	dev_time = magma_sync_wtime(queue);
	magma_dsetmatrix(targets, targets, IF1_h, targets, IF1_d, size, targets, queue);
	tTransfer += magma_sync_wtime(queue) - dev_time;


	dev_time = magma_sync_wtime(queue);
	magma_dgemm(MagmaNoTrans, MagmaNoTrans, linessamples, targets, targets, alpha, yy_d, size, linessamples, IF1_d, size, targets, beta, X_d, size, linessamples, queue);
	tExeGPU += magma_sync_wtime(queue) - dev_time;
	//printf("dgemm yy*IF1: %f\n",dev_time);


	dev_time = magma_sync_wtime(queue);
	magma_dgetmatrix(linessamples,targets, X_d, size, linessamples, abundancias_h, linessamples, queue);
	tTransfer += magma_sync_wtime(queue) - dev_time;

//-------------
//for(int z = 0; z < targets; z++) printf("%f ",abundancias_h[z]);printf("\n%f\n",abundancias_h[linessamples*targets-1]);
//-------------
	dev_time = magma_sync_wtime(queue);	
	for (k = 0; k < targets; k++){
		auxk = Aux_h[k];
		for (ii = 0; ii < linessamples;ii++){
			abundancias_h[k*linessamples+ii] = abundancias_h[k*linessamples+ii] + auxk;	
		}
	}
	tExeCPU += magma_sync_wtime(queue) - dev_time;

	


	if(filename != NULL){

		magma_finalize();

		printf("\nTotal INIT:	\t%.3f (seconds)\n", t1Init-t0Init);
		double tTotal = (t1Init-t0Init)	+ tTransfer + tExeCPU + tExeGPU;
		double tExe =  tTransfer + tExeCPU + tExeGPU;
		printf("\nTotal LSU:	\t%.3f (seconds)\n Transfer:    \t\t%.3f\t(%2.1f%) \n Execution(CPU): \t%.3f\t(%2.1f%)\n Execution(GPU): \t%2.3f\t(%.1f%)\n", tExe, tTransfer, (tTransfer*100)/tExe, tExeCPU, (tExeCPU*100)/tExe, tExeGPU, (tExeGPU*100)/tExe);
		printf("\nTotal:	\t%.3f (seconds)\n\n",tTotal);

	
		char results_filename[MAXCAD];
		strcpy(results_filename, filename);
		strcat(results_filename, "Results.hdr");
		writeHeader(results_filename, samples, lines, targets);


		strcpy(results_filename, filename);
		strcat(results_filename, "Results.bsq");
		writeResult(abundancias_h, results_filename, samples, lines, targets);
	}


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




