#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define MAXLINE 200
#define MAXCAD 90


//Viena Opencl Libraries
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/tools/timer.hpp"
#include "viennacl/linalg/lu.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/linalg/sum.hpp"

#ifndef VIENNACL_WITH_OPENCL
	#define VIENNACL_WITH_OPENCL 
#endif

//uBlas Headers
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
//#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
//#include <boost/numeric/ublas/matrix_proxy.hpp>
//#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
//#include <boost/numeric/ublas/io.hpp>





void cleanString(char *cadena, char *out)
{
    int i,j;
    for( i = j = 0; cadena[i] != 0;++i)
    {
        if(isalnum(cadena[i])||cadena[i]=='{'||cadena[i]=='.'||cadena[i]==',')
        {
            out[j]=cadena[i];
            j++;
        }
    }
    for( i = j; out[i] != 0;++i)
        out[j]=0;
}

int readHeader1(char* filename, int *lines, int *samples, int *bands, int *dataType,
		char* interleave, int *byteOrder, char* waveUnit)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char value [MAXLINE] = "";

    if ((fp=fopen(filename,"rt"))!=NULL)
    {
        fseek(fp,0L,SEEK_SET);
        while(fgets(line, MAXLINE, fp)!='\0')
        {
            //Samples
            if(strstr(line, "samples")!=NULL && samples !=NULL)
            {
                cleanString(strstr(line, "="),value);
                *samples = atoi(value);
            }

            //Lines
            if(strstr(line, "lines")!=NULL && lines !=NULL)
            {
                cleanString(strstr(line, "="),value);
                *lines = atoi(value);
            }

            //Bands
            if(strstr(line, "bands")!=NULL && bands !=NULL)
            {
                cleanString(strstr(line, "="),value);
                *bands = atoi(value);
            }

            //Interleave
            if(strstr(line, "interleave")!=NULL && interleave !=NULL)
            {
                cleanString(strstr(line, "="),value);
                strcpy(interleave,value);
            }

            //Data Type
            if(strstr(line, "data type")!=NULL && dataType !=NULL)
            {
                cleanString(strstr(line, "="),value);
                *dataType = atoi(value);
            }

            //Byte Order
            if(strstr(line, "byte order")!=NULL && byteOrder !=NULL)
            {
                cleanString(strstr(line, "="),value);
                *byteOrder = atoi(value);
            }

            //Wavelength Unit
            if(strstr(line, "wavelength unit")!=NULL && waveUnit !=NULL)
            {
                cleanString(strstr(line, "="),value);
                strcpy(waveUnit,value);
            }

        }
        fclose(fp);
        return 0;
    }
    else
    	return -2; //No file found
}

int readHeader2(char* filename, double* wavelength)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char value [MAXLINE] = "";
    char *info;

    if ((fp=fopen(filename,"rt"))!=NULL)
    {
        fseek(fp,0L,SEEK_SET);
        while(fgets(line, MAXLINE, fp)!='\0')
        {
            //Wavelength
            if(strstr(line, "wavelength =")!=NULL && wavelength !=NULL)
            {
                char strAll[100000]=" ";
                char *pch;
                int cont = 0;
                do
                {
                    info = fgets(line, 200, fp);
                    cleanString(line,value);
                    strcat(strAll,value);
                } while(strstr(line, "}")==NULL);

                pch = strtok(strAll,",");

                while (pch != NULL)
                {
                    wavelength[cont]= atof(pch);
                    pch = strtok (NULL, ",");
                    cont++;
                }
            }

		}
		fclose(fp);
		return 0;
	}
	else
		return -2; //No file found
}

int loadImage(char* filename, boost::numeric::ublas::matrix<double>& image, int lines, int samples, int bands, int dataType, char* interleave){

    FILE *fp;
    short int *tipo_short_int;
    float *tipo_float;
    double * tipo_double;
    int i, j, k, op;
    long int lines_samples = lines*samples;
    size_t size;


    if ((fp=fopen(filename,"rb"))!=NULL){

        fseek(fp,0L,SEEK_SET);
        tipo_double = (double*)malloc(lines_samples*bands*sizeof(double));
        switch(dataType){
            case 2:
                tipo_short_int = (short int*)malloc(lines_samples*bands*sizeof(short int));
                size = fread(tipo_short_int,1,(sizeof(short int)*lines_samples*bands),fp);
                for(i=0; i<lines_samples * bands; i++)
                	tipo_double[i]=(double)tipo_short_int[i];
                free(tipo_short_int);
                break;

            case 4:
            	tipo_float = (float*)malloc(lines_samples*bands*sizeof(float));
                size = fread(tipo_float,1,(sizeof(float)*lines_samples*bands),fp);
                for(i=0; i<lines_samples * bands; i++)
                	tipo_double[i]=(double)tipo_float[i];
                free(tipo_float);
                break;

            case 5:
                size = fread(tipo_double,1,(sizeof(double)*lines_samples*bands),fp);
                break;

        }
        fclose(fp);

        if(interleave == NULL)
        	op = 0;
        else{
		    if(strcmp(interleave, "bsq") == 0) op = 0;
		    if(strcmp(interleave, "bip") == 0) op = 1;
		    if(strcmp(interleave, "bil") == 0) op = 2;
        }


        switch(op){
		    case 0://mis archivos son bsq asi que solo he modificado este bucle!
			    for(i=0; i<lines*samples; i++)
					for(j=0; j<bands; j++)
				    	image(i,j) = tipo_double[i + lines*samples*j];
			    break;

		    case 1:
			    for(i=0; i<bands; i++)
				    for(j=0; j<lines*samples; j++)
					    //image[i*lines*samples + j] = tipo_double[j*bands + i];
			    break;

		    case 2:
			    for(i=0; i<lines; i++)
				    for(j=0; j<bands; j++)
					    for(k=0; k<samples; k++)
						    //image[j*lines*samples + (i*samples + k)] = tipo_double[i*bands*samples + (j*samples + k)];
			    break;
        }
		//std::cout << image << std::endl;
        free(tipo_double);
        return 0;
    }
    return -2;
}

int writeResult(boost::numeric::ublas::matrix<double>& image, const char* filename, int lines, int samples, int bands, int dataType, char* interleave)
{

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

int writeHeader(char* filename, int lines, int samples, int bands, int dataType,
		char* interleave, int byteOrder, char* waveUnit, double* wavelength)
{
    FILE *fp;
    if ((fp=fopen(filename,"wt"))!=NULL)
    {
		fseek(fp,0L,SEEK_SET);
		fprintf(fp,"ENVI\ndescription = {\nExported from MATLAB}\n");
		if(samples != 0) fprintf(fp,"samples = %d", samples);
		if(lines != 0) fprintf(fp,"\nlines   = %d", lines);
		if(bands != 0) fprintf(fp,"\nbands   = %d", bands);
		if(dataType != 0) fprintf(fp,"\ndata type = %d", dataType);
		if(interleave != NULL) fprintf(fp,"\ninterleave = %s", interleave);
		if(byteOrder != 0) fprintf(fp,"\nbyte order = %d", byteOrder);
		if(waveUnit != NULL) fprintf(fp,"\nwavelength units = %s", waveUnit);
		if(waveUnit != NULL){
			fprintf(fp,"\nwavelength = {\n");
			for(int i=0; i<bands; i++)
			{
				if(i==0) fprintf(fp, "%f", wavelength[i]);
				else
					if(i%3 == 0) fprintf(fp, ", %f\n", wavelength[i]);
					else fprintf(fp, ", %f", wavelength[i]);
			}
			fprintf(fp,"}");
		}
		fclose(fp);
		return 0;
    }
    return -3;
}
	
double get_time(){
	static struct timeval tv0;
	double time_, time;

	gettimeofday(&tv0,(struct timezone*)0);
	time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
	time = time_/1000000;
	return(time);
}


int main(int argc, char* argv[])
{

	/*
	 * PARAMETERS
	 *
	 * argv[1]: Input image file
	 * argv[2]: Input endmembers file
	 * argv[3]: Output abundances file
	 * argv[4]: Platform: 0 -> CPU, 1 -> GPU, 2 -> Accelerator
	 * */
	if(argc !=  5)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Parameters are not correct.\n");
		printf("./SCLSU [Image Filename] [Endmembers file] [Output Result File] [0|1|2]\n");
		printf("argv[4]: Platform: 0 -> CPU, 1 -> GPU, 2 -> Accelerator\n");
		fflush(stdout);
		exit(-1);
	}
	int DeviceSelected = atoi(argv[4]);
	//READ IMAGE
	char header_filename[MAXCAD];
	char imagen_filename[MAXCAD];
	strcpy(header_filename, argv[1]);
	strcat(header_filename, ".hdr");

	int lines = 0, samples= 0, bands= 0, dataType= 0, byteOrder = 0;
	char *interleave, *waveUnit;
	interleave = (char*)malloc(MAXCAD*sizeof(char));
	waveUnit = (char*)malloc(MAXCAD*sizeof(char));

	// Load image
	int error = readHeader1(header_filename, &lines, &samples, &bands, &dataType, interleave, &byteOrder, waveUnit);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}
	printf("Imagen -> Lineas: %d, Muestras: %d, bandas: %d, Tipo de datos: %d\n",lines,samples,bands,dataType);//<--
	double* wavelength = (double*)malloc(bands*sizeof(double));
	strcpy(header_filename,argv[1]); // Second parameter: Header file:
	strcat(header_filename,".hdr");
	error = readHeader2(header_filename, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error reading header file: %s.", header_filename);
		fflush(stdout);
		exit(-1);
	}

	strcpy(imagen_filename, argv[1]);
	strcat(imagen_filename, ".bsq");

	boost::numeric::ublas::matrix<double> imagen_Host(lines*samples, bands);
	error = loadImage(imagen_filename, imagen_Host, lines, samples, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error reading image file: %s.(%d)\n", argv[1],error);
		fflush(stdout);
		exit(-1);
	}
//-----------------------
	//READ ENDMEMBERS
	int samplesE, targets, bandsEnd;
	char *interleaveE;
	interleaveE = (char*)malloc(MAXCAD*sizeof(char));

	strcpy(header_filename, argv[2]);
	strcat(header_filename, ".hdr");
	error = readHeader1(header_filename, &targets, &samplesE, &bandsEnd, &dataType, interleaveE, NULL, NULL);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error reading endmembers header file: %s.", header_filename);
		fflush(stdout);
		return error;
	}
	printf("Endmembers -> targets: %d, Muestras: %d, bandas: %d, Tipo de datos: %d\n",targets,samplesE,bandsEnd,dataType);//<--
	boost::numeric::ublas::matrix<double> endmember_Host(targets, bandsEnd);
	error = loadImage(argv[2], endmember_Host, targets, samplesE, bandsEnd, dataType, interleaveE);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error reading endmembers file: %s.", argv[2]);
		fflush(stdout);
		return error;
	}

	//START CLOCK***************************************
	//clock_t start, end;
	double start = get_time();
	//**************************************************

	int i, j, one = 1;
	/*double alpha = 1, beta = 0;
	double *Et_E = (double*)malloc(targets*targets*sizeof(double));*/

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
	boost::numeric::ublas::vector<double> PIXEL_Host(bandsEnd);
	double Y_Host;

    	//Device
	viennacl::matrix<double, viennacl::column_major> endmember_Device(targets, bandsEnd, ctx);
    	viennacl::matrix<double> EtE_Device(targets, targets);
	viennacl::vector<double> ONE_Device = viennacl::scalar_vector<double>(targets, 1);
    	viennacl::vector<double> AUX_Device(targets);
   	viennacl::vector<double> AUX2_Device(targets);
	viennacl::scalar<double> Y_Device = double(1.0);
	viennacl::matrix<double> I_Device(targets, targets);
	viennacl::matrix<double> A_Device(targets, targets);
	viennacl::matrix<double> B_Device(bandsEnd, targets);
	viennacl::vector<double> PIXEL_Device(bandsEnd);



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
		for(j = 0; j < bandsEnd; j++) PIXEL_Host(j) = imagen_Host(i,j);
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

	strcpy(header_filename, argv[3]);
	strcat(header_filename, ".hdr");
	error = writeHeader(header_filename, lines,samples, targets, 5, interleave, 0, NULL, NULL);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error writing endmembers header file: %s.", header_filename);
		fflush(stdout);
		return error;
	}

	error = writeResult(imagen_Host,argv[3],lines,samples, targets, 5, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error writing endmembers file: %s.", argv[3]);
		fflush(stdout);
		return error;
	}

	//FREE MEMORY***************************************

	free(wavelength);
	free(interleaveE);
	free(interleave);
	free(waveUnit);

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




	return 0;
}






























