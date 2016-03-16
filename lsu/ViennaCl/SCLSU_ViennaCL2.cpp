#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
//#include <CL/cl.h>

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




/*
 * Author: Jorge Sevilla Cedillo
 * Centre: Universidad de Extremadura
 * */
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

/*
 * Author: Jorge Sevilla Cedillo
 * Centre: Universidad de Extremadura
 * */
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

/*
 * Author: Jorge Sevilla Cedillo
 * Centre: Universidad de Extremadura
 * */
int readHeader2(char* filename, double* wavelength)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char value [MAXLINE] = "";

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
                    fgets(line, 200, fp);
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


/*
 * Author: Jorge Sevilla Cedillo
 * Centre: Universidad de Extremadura
 * */
int loadImage(char* filename, double* image, int lines, int samples, int bands, int dataType, char* interleave)
{

    FILE *fp;
    short int *tipo_short_int;
    float *tipo_float;
    double * tipo_double;
    int i, j, k, op;
    long int lines_samples = lines*samples;


    if ((fp=fopen(filename,"rb"))!=NULL)
    {

        fseek(fp,0L,SEEK_SET);
        tipo_double = (double*)malloc(lines_samples*bands*sizeof(double));
        switch(dataType)
        {
            case 2:
                tipo_short_int = (short int*)malloc(lines_samples*bands*sizeof(short int));
                fread(tipo_short_int,1,(sizeof(short int)*lines_samples*bands),fp);
                for(i=0; i<lines_samples * bands; i++)
                	tipo_double[i]=(double)tipo_short_int[i];
                free(tipo_short_int);
                break;

            case 4:
            	tipo_float = (float*)malloc(lines_samples*bands*sizeof(float));
                fread(tipo_float,1,(sizeof(float)*lines_samples*bands),fp);
                for(i=0; i<lines_samples * bands; i++)
                	tipo_double[i]=(double)tipo_float[i];
                free(tipo_float);
                break;

            case 5:
                fread(tipo_double,1,(sizeof(double)*lines_samples*bands),fp);
                break;

        }
        fclose(fp);

        if(interleave == NULL)
        	op = 0;
        else
        {
        	if(strcmp(interleave, "bsq") == 0) op = 0;
        	if(strcmp(interleave, "bip") == 0) op = 1;
        	if(strcmp(interleave, "bil") == 0) op = 2;
        }


        switch(op)
        {
			case 0:
				for(i=0; i<lines*samples*bands; i++)
					image[i] = tipo_double[i];
				break;

			case 1:
				for(i=0; i<bands; i++)
					for(j=0; j<lines*samples; j++)
						image[i*lines*samples + j] = tipo_double[j*bands + i];
				break;

			case 2:
				for(i=0; i<lines; i++)
					for(j=0; j<bands; j++)
						for(k=0; k<samples; k++)
							image[j*lines*samples + (i*samples + k)] = tipo_double[i*bands*samples + (j*samples + k)];
				break;
        }
        free(tipo_double);
        return 0;
    }
    return -2;
}


/*
 * Author: Luis Ignacio Jimenez
 * Centre: Universidad de Extremadura
 * */
int writeResult(double *image, const char* filename, int lines, int samples, int bands, int dataType, char* interleave)
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
				for(i=0; i<lines*samples*bands; i++)
					imageSI[i] = (short int)image[i];
				break;

			case 1:
				for(i=0; i<bands; i++)
					for(j=0; j<lines*samples; j++)
						imageSI[j*bands + i] = (short int)image[i*lines*samples + j];
				break;

			case 2:
				for(i=0; i<lines; i++)
					for(j=0; j<bands; j++)
						for(k=0; k<samples; k++)
							imageSI[i*bands*samples + (j*samples + k)] = (short int)image[j*lines*samples + (i*samples + k)];
				break;
        }

	}

	if(dataType == 4)
	{
		imageF = (float*)malloc(lines*samples*bands*sizeof(float));
        switch(op)
        {
			case 0:
				for(i=0; i<lines*samples*bands; i++)
					imageF[i] = (float)image[i];
				break;

			case 1:
				for(i=0; i<bands; i++)
					for(j=0; j<lines*samples; j++)
						imageF[j*bands + i] = (float)image[i*lines*samples + j];
				break;

			case 2:
				for(i=0; i<lines; i++)
					for(j=0; j<bands; j++)
						for(k=0; k<samples; k++)
							imageF[i*bands*samples + (j*samples + k)] = (float)image[j*lines*samples + (i*samples + k)];
				break;
        }
	}

	if(dataType == 5)
	{
		imageD = (double*)malloc(lines*samples*bands*sizeof(double));
        switch(op)
        {
			case 0:
				for(i=0; i<lines*samples*bands; i++)
					imageD[i] = image[i];
				break;

			case 1:
				for(i=0; i<bands; i++)
					for(j=0; j<lines*samples; j++)
						imageD[j*bands + i] = image[i*lines*samples + j];
				break;

			case 2:
				for(i=0; i<lines; i++)
					for(j=0; j<bands; j++)
						for(k=0; k<samples; k++)
							imageD[i*bands*samples + (j*samples + k)] = image[j*lines*samples + (i*samples + k)];
				break;
        }
	}

    FILE *fp;
    if ((fp=fopen(filename,"wb"))!=NULL)
    {
        fseek(fp,0L,SEEK_SET);
	    switch(dataType)
	    {
	    case 2: fwrite(imageSI,1,(lines*samples*bands * sizeof(short int)),fp); free(imageSI); break;
	    case 4: fwrite(imageF,1,(lines*samples*bands * sizeof(float)),fp); free(imageF); break;
	    case 5: fwrite(imageD,1,(lines*samples*bands * sizeof(double)),fp); free(imageD); break;
	    }
	    fclose(fp);


	    return 0;
    }

    return -3;
}

/*
 * Author: Luis Ignacio Jimenez
 * Centre: Universidad de Extremadura
 * */
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
		if(waveUnit != NULL)
		{
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

void exitOnFail(cl_int status, const char* message){
	if (CL_SUCCESS != status){
		printf("error: %s\n", message);
		exit(-1);
	}
}


/*
 * Author: Luis Ignacio Jimenez Gil
 * Centre: Universidad de Extremadura
 * */
int main(int argc, char* argv[])
{

	/*
	 * PARAMETERS
	 *
	 * argv[1]: Input image file
	 * argv[2]: Input endmembers file
	 * argv[3]: Output abundances file
	 * */
	if(argc !=  5)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Parameters are not correct.\n");
		printf("./SCLSU [Image Filename] [Endmembers file] [Output Result File] [0|1|2]\n");
		printf("argv[4]: Platform: 0 -> CPU, 1 -> GPU, 2 -> Accelerator\n");
		fflush(stdout);
		exit(-1);
	}

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
	
	double *image = (double*)malloc(lines*samples*bands*sizeof(double));
	error = loadImage(imagen_filename, image, lines, samples, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error reading image file: %s.", argv[1]);
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
	double *endmembers = (double*)malloc(targets*bandsEnd*sizeof(double));
	error = loadImage(argv[2], endmembers, targets, samplesE, bandsEnd, dataType, interleaveE);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error reading endmembers file: %s.", argv[2]);
		fflush(stdout);
		return error;
	}
	int deviceSelected = atoi(argv[4]), ok=0,i,j;

	cl_int status;
	cl_context clContext;
	cl_command_queue clCommandQue;
	cl_program clProgram;
	cl_kernel clKernel;

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
		return 1;
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

	std::vector<cl_device_id> device_id_array;
	//get all available devices
	viennacl::ocl::platform pf;
	std::cout << "Platform info: " << pf.info() << std::endl;
	std::vector<viennacl::ocl::device> devices = pf.devices(CL_DEVICE_TYPE_DEFAULT);
	std::cout << devices[0].name() << std::endl;
	std::cout << "Number of devices for custom context: " << devices.size() << std::endl;
	//set up context using all found devices:
	for (std::size_t i=0; i<devices.size(); ++i)
	{
	device_id_array.push_back(devices[i].id());
	}
	std::cout << "Creating context..." << std::endl;
	cl_int err;
	cl_context my_context = clCreateContext(0, cl_uint(device_id_array.size()), &(device_id_array[0]), NULL, NULL, &err);
	VIENNACL_ERR_CHECK(err);

	//START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************
	clContext = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &status);//Context
	exitOnFail( status, "clCreateContext" );
	
	clCommandQue = clCreateCommandQueue(clContext, deviceID, CL_QUEUE_PROFILING_ENABLE, &status);//Queue
  	exitOnFail(status, "create command queue");

	cl_mem image_Host = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, lines * samples * bands * sizeof(double), image, &status);
	VIENNACL_ERR_CHECK(status);
	cl_mem endmembers_Host = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, targets * bandsEnd * sizeof(double), endmembers, &status);
	VIENNACL_ERR_CHECK(status);

	std::vector<cl_command_queue> queues(devices.size());
	for (std::size_t i=0; i<devices.size(); ++i){
		queues[i] = clCreateCommandQueue(my_context, devices[i].id(), 0, &err);
		VIENNACL_ERR_CHECK(err);
	}

	viennacl::ocl::setup_context(0, my_context, device_id_array, queues);
	viennacl::ocl::switch_context(0); //activate the new context (only mandatory with context-id not equal to zero)
	std::cout << "Existing context: " << my_context << std::endl;
	std::cout << "ViennaCL uses context: " << viennacl::ocl::current_context().handle().get() << std::endl;

	viennacl::tools::timer timer;
	timer.start();

    	//Host
  	boost::numeric::ublas::matrix<double> EtE_Host_aux(targets, targets);
  	boost::numeric::ublas::matrix<double> EtE_Host(targets, targets);
	boost::numeric::ublas::permutation_matrix<size_t> permutation_Host(targets);
	boost::numeric::ublas::vector<double> AUX2_Host(targets);
	boost::numeric::ublas::matrix<double> I_Host(targets, targets);
	double Y_Host;

    	//Device
	viennacl::matrix<double, viennacl::column_major> endmember_Device(endmembers_Host, targets, bandsEnd);
    	viennacl::matrix<double> EtE_Device(targets, targets);
	viennacl::vector<double> ONE_Device = viennacl::scalar_vector<double>(targets, 1);
    	viennacl::vector<double> AUX_Device(targets);
   	viennacl::vector<double> AUX2_Device(targets);
	viennacl::scalar<double> Y_Device = double(1.0);
	viennacl::matrix<double> I_Device(targets, targets);
	viennacl::matrix<double> A_Device(targets, targets);
	viennacl::matrix<double> B_Device(bandsEnd, targets);
	viennacl::vector<double> PIXEL_Device(bandsEnd);
    	viennacl::matrix<double> imagen_Device(image_Host, lines*samples, bands);

	double t0 = timer.get();
	EtE_Device = viennacl::linalg::prod(endmember_Device, viennacl::trans(endmember_Device));
	double t1 = timer.get();
	printf("Time Et_E: %f\n",t1-t0);
	t1 = timer.get();

	viennacl::copy(EtE_Device, EtE_Host_aux);
	double t3 = timer.get();
	//std::cout << EtE_Host_aux << std::endl;

	boost::numeric::ublas::lu_factorize(EtE_Host_aux, permutation_Host);
	boost::numeric::ublas::lu_substitute(EtE_Host_aux, permutation_Host, EtE_Host);
	double t4 = timer.get();
//std::cout << EtE_Host << std::endl;

	viennacl::copy(EtE_Host, EtE_Device);
	double t5 = timer.get();

	printf("Tiempo EtE (inversa->uBlas): %f \n", t4-t3);
	printf("Tiempo EtE: %f \n", t5-t0);//Tiempo 1: 0.004420
	printf("Tiempo EtE(prod): %f\n-------\n",t1-t0);



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
	
	EtE_Device = viennacl::scalar_matrix<double>(targets, targets, 2);//<----------------------------borrarlo solo para pruebas
	t0 = timer.get();
	AUX_Device = viennacl::linalg::prod(EtE_Device, ONE_Device);//Et_E
	t1 = timer.get();
	printf("Tiempo Aux_device: %f \n", t1-t0);

	t0 = timer.get();
	//Y_Device = viennacl::linalg::sum(AUX_Device);
	Y_Device = 1 / Y_Device;
	Y_Host = Y_Device; printf("Y_Host: %f\n",Y_Host);
	AUX_Device = viennacl::scalar_vector<double>(targets, Y_Device);
	t1 = timer.get();
	printf("Tiempo Y_device: %f \n", t1-t0);
	
	//dgemm_("N", "N", &one, &targets, &targets, &alpha, AUX, &one, Et_E, &targets, &beta, AUX2, &one);
	AUX2_Device = viennacl::linalg::prod(EtE_Device, AUX_Device);
	

	t0 = timer.get();
	viennacl::copy(AUX2_Device, AUX2_Host);//me interesa mas traermelo al completo, modificarlo y volverlo a llevar
	for(i=0; i<targets; i++)
		for(j=0; j<targets; j++)
			if(i == j)
				I_Host(i,j) = 1 - AUX2_Host(j);
			else
				I_Host(i,j) = -AUX2_Host(j);
	viennacl::copy(I_Host, I_Device);
	t1 = timer.get();
	printf("Tiempo I_Host: %f \n", t1-t0);


	t0 = timer.get();
	A_Device = viennacl::linalg::prod(I_Device, EtE_Device);
	
	//A_Device = viennacl::identity_matrix<double>(targets);//---

	B_Device = viennacl::linalg::prod( viennacl::trans(endmember_Device),A_Device);//ojo que esta alreves 188*19

	//std::cout << B_Device << std::endl;

	//dgemm_("N", "N", &targets, &one, &targets, &alpha, Et_E, &targets, ONE, &targets, &beta, AUX, &targets);
	//for(i=0; i<targets; i++) AUX[i] *= Y;
	AUX_Device = viennacl::linalg::prod(EtE_Device, ONE_Device);
	AUX_Device = Y_Device * AUX_Device;
	t1 = timer.get();
	printf("Tiempo A - B - I time: %f \n", t1-t0);
	/*viennacl::copy(AUX_Device, AUX2_Device);
	std::cout << AUX2_Device << std::endl;*/

	/*double* ABUN = (double*)calloc(lines*samples*targets, sizeof(double));
	double* PIXEL = (double*)calloc(bands, sizeof(double));

	for(i=0; i<lines*samples; i++)
	{
		for(j=0; j<bands; j++) PIXEL[j] = image[j*lines*samples+i];//extrae el pixel iesimo con todas sus bandas
	//	dgemm_("N", "N", &targets, &one, &bands, &alpha, B, &targets, PIXEL, &bands, &beta, AUX2, &targets);

		for(j=0; j<targets; j++) ABUN[j*lines*samples + i] = AUX2[j] + AUX[j];
	}*/

	
	t0 = timer.get();
	for(i = 0; i < lines*samples; i++){
		PIXEL_Device = viennacl::row(imagen_Device,i);
		AUX2_Device = viennacl::linalg::prod(viennacl::trans(B_Device), PIXEL_Device);//B * Pixel
		//if(i == 0) std::cout << AUX2_Device << std::endl; 
		AUX2_Device = AUX2_Device + AUX_Device;
		viennacl::copy(AUX2_Device, AUX2_Host);
		//copiar el array aux2_host en imagen host//<---- aqui me quede por hacer, cambiar el write
		//for(j = 0; j < bandsEnd; j++) imagen_Host(i,j) = AUX2_Host(j);//opcion mas directa como la de extraer row?
		if(i == 0) t3 = timer.get();
	}
	t1 = timer.get();
	printf("Tiempo image modifications: %f \n", t1-t0);
	printf("Tiempo image modifications 1 iter: %f \n",t3-t0);
	

	viennacl::backend::finish();

	

	

	//END CLOCK*****************************************
	end = clock();
	printf("Iterative SCLSU: %f segundos\n", (double)(end - start) / CLOCKS_PER_SEC);
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

	/*error = writeResult(ABUN,argv[3],lines,samples, targets, 5, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error writing endmembers file: %s.", argv[3]);
		fflush(stdout);
		return error;
	}*/

	//FREE MEMORY***************************************
	/*free(Et_E);
	free(ipiv);
	free(work);
	free(ONE);
	free(AUX);
	free(AUX2);
	free(I);
	free(A);
	free(B);
	free(PIXEL);
	free(ABUN);*/
	free(wavelength);
	free(image);
	free(endmembers);
	free(interleaveE);
	free(interleave);
	free(waveUnit);

	return 0;
}






























