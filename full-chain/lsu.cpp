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
		return 0; 
	}



    return -3;
}



void lsu_gpu_v(float *imagen, float *endmembers, int DeviceSelected, int bandas, int targets, int lines, int samples, char *filename){
	
	int i, j, one = 1;
	int sampleLines = lines * samples;
	boost::numeric::ublas::matrix<float> imagen_Host(sampleLines, bandas);
	boost::numeric::ublas::matrix<float> endmember_Host(targets, bandas);
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
































