#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <iostream>
#include <math.h>

//#include <clBLAS.h>
//#include "cblas.h"
#include <magma.h>
#include <magma_lapack.h>


#define MAXLINE 200
#define MAXCAD 90



int writeHeader(char* filename, int lines, int samples, int bands, int dataType, char* interleave, int byteOrder, char* waveUnit, double* wavelength);
int readHeader1(char* filename, int *lines, int *samples, int *bands, int *dataType, char* interleave, int *byteOrder, char* waveUnit);
int readHeader2(char* filename, double* wavelength);
int loadImage(char* filename, double* image, int lines, int samples, int bands, int dataType, char* interleave);
int writeResult(double *image, const char* filename, int lines, int samples, int bands, int dataType, char* interleave);
int writeHeader(char* filename, int lines, int samples, int bands, int dataType, char* interleave, int byteOrder, char* waveUnit, double* wavelength);
void exitOnFail(cl_int status, const char* message);

#define MALLOC_HOST( ptr, type, size )                                     \
    if ( MAGMA_SUCCESS !=                                                  \
            magma_malloc_cpu( (void**) &ptr, (size)*sizeof(type) )) {      \
        fprintf( stderr, "!!!! magma_malloc_cpu failed for: %s\n", #ptr ); \
        magma_finalize();                                                  \
        exit(-1);                                                          \
    }

#define MALLOC_DEVICE( ptr, type, size )                               \
if ( MAGMA_SUCCESS !=                                                  \
        magma_malloc( &ptr, (size)*sizeof(type) )) {                   \
    fprintf( stderr, "!!!! magma_malloc failed for: %s\n", #ptr );     \
    magma_finalize();                                                  \
    exit(-1);                                                          \
}


void SCLSU_Magma(char *imagenIn, char *endmembersIn, char *imagenOut, int deviceSelected){

	//READ IMAGE
	char header_filename[MAXCAD];
	strcpy(header_filename, imagenIn);
	strcat(header_filename, ".hdr");

	char imagen_filename[MAXCAD];
	strcpy(imagen_filename, imagenIn);
	strcat(imagen_filename, ".bsq");


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
	error = readHeader2(header_filename, wavelength);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error reading header file: %s\n", header_filename);
		fflush(stdout);
		exit(-1);
	}

	double *image_Host;
	MALLOC_HOST(image_Host, double, lines*samples*bands)
	error = loadImage(imagen_filename, image_Host, lines, samples, bands, dataType, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error reading image file: %s\n", imagen_filename);
		fflush(stdout);
		exit(-1);
	}
//-----------------------
	//READ ENDMEMBERS
	int samplesE = 0, targets = 0, bandsEnd = 0;
	char *interleaveE;
	interleaveE = (char*)malloc(MAXCAD*sizeof(char));

	strcpy(header_filename, endmembersIn);
	strcat(header_filename, ".hdr");
	strcpy(imagen_filename, endmembersIn);
	strcat(imagen_filename, ".bsq");

	error = readHeader1(header_filename, &targets, &samplesE, &bandsEnd, &dataType, interleaveE, NULL, NULL);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error reading endmembers header file: %s\n", header_filename);
		fflush(stdout);
		exit(-1);
	}
	printf("Endmembers -> targets: %d, Muestras: %d, bandas: %d, Tipo de datos: %d\n",targets,samplesE,bandsEnd,dataType);//<--
	double *endmember_Host;
	MALLOC_HOST(endmember_Host, double, targets*bandsEnd)
	error = loadImage(imagen_filename, endmember_Host, targets, samplesE, bandsEnd, dataType, interleaveE);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error reading endmembers file: %s\n", imagen_filename);
		fflush(stdout);
		exit(-1);
	}

	//--------------------------------OPENCL-------------------------------------------------------------//
	// return code used by OpenCL API
    	cl_int status;
	unsigned int ok = 0, i, j;

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
	//cl_platform_id platformID;
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
		exit(-1);
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
//-------------
	real_Double_t dev_time, total_time;
	double alpha = 1.0;
	double beta  = 0.0;
	size_t size = 0;
	int info;
	int lwork = targets;//redundante ¿?
	double Y = 0;
	int one = 1;
	//DEVICE
	magmaDouble_ptr endmember_Device, image_Device, EtE_Device, work_Device, one_Device, aux_Device, aux2_Device, I_Device, A_Device, B_Device, pixel_Device;
	MALLOC_DEVICE(endmember_Device, double, targets*bandsEnd)
	MALLOC_DEVICE(image_Device, double, lines*samples*bands)
	MALLOC_DEVICE(EtE_Device, double, targets*targets)
	MALLOC_DEVICE(work_Device, double, lwork)//probar memoria pinned <- not supported
	MALLOC_DEVICE(one_Device, double, targets)
	MALLOC_DEVICE(aux_Device, double, targets)
	MALLOC_DEVICE(aux2_Device, double, targets)
	MALLOC_DEVICE(I_Device, double, targets*targets)
	MALLOC_DEVICE(A_Device, double, targets*targets)
	MALLOC_DEVICE(B_Device, double, targets*bandsEnd)
	MALLOC_DEVICE(pixel_Device, double, bands)


	//HOST
	double *EtE_Host, *one_Host, *aux_Host, *aux2_Host, *I_Host, *pixel_Host, *abundancias_Host;
	magma_int_t *ipiv_Host;
	MALLOC_HOST(EtE_Host, double, targets*targets)
	MALLOC_HOST(ipiv_Host, magma_int_t, targets)//pivote de la inversion
	MALLOC_HOST(one_Host, double, targets)
	MALLOC_HOST(aux_Host, double, targets)
	MALLOC_HOST(aux2_Host, double, targets)
	MALLOC_HOST(I_Host, double, targets*targets)
	MALLOC_HOST(pixel_Host, double, bands)
	MALLOC_HOST(abundancias_Host, double, lines*samples*targets)


	


 	//magma_roundup -> ((m + 31)/32)*32//podria mejorar ligeramente, segun los testing

	total_time = magma_sync_wtime(queue);
	magma_dsetmatrix(targets, bandsEnd, endmember_Host, targets, endmember_Device, size, targets, queue);
	dev_time = magma_sync_wtime(queue) - total_time;
	printf("EtE_transfer host -> device: %f\n",dev_time);

	dev_time = magma_sync_wtime(queue);
	magma_dgemm(MagmaNoTrans, MagmaTrans, targets, targets, bandsEnd, alpha, endmember_Device, size, targets, endmember_Device, size, targets, beta,  EtE_Device, size, targets, queue);
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
	magma_dgemm(MagmaNoTrans, MagmaNoTrans, targets, bandsEnd, targets, alpha, A_Device, size, targets, endmember_Device, size, targets, beta, B_Device, size, targets, queue);
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
		for(int n = 0; n < bands; n++) pixel_Host[n] = image_Host[n*lines*samples+m];
		magma_dsetmatrix(bands, one, pixel_Host, bands, pixel_Device, size, bands, queue);
		magma_dgemm(MagmaNoTrans, MagmaNoTrans, targets, one, bands, alpha, B_Device, size, targets, pixel_Device, size, bands, beta, aux2_Device, size, targets, queue);

		//magma_daxpy(targets, alpha, aux_Device, one, aux2_Device, one, queue);//no esta disponible en la version 1.3 <.<!
		magma_dgetmatrix(targets, one, aux2_Device, size, targets, aux2_Host, targets, queue);
		for(int n = 0; n < targets; n++) abundancias_Host[n*lines*samples + m] = aux2_Host[n]+aux_Host[n];
		
		//if(m == 0) for(int n = 0; n < targets; n++) printf("%f ", aux2_Host[n]+aux_Host[n]);
	}
	dev_time = magma_sync_wtime(queue) - dev_time;
	printf("abundancias_solucion: %f\n",dev_time);
	



	total_time = magma_sync_wtime(queue) - total_time;
	printf("Total Time: %f\n",total_time);

	magma_finalize();	

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

	free(waveUnit);
	free(wavelength);
	free(interleave);
	free(interleaveE);


}




// ------------------------------------------------------------
int main( int argc, char** argv )
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
		printf("EXECUTION ERROR SCLSU Iterative: Parameters are not correct.");
		printf("./SCLSU [Image Filename] [Endmembers file] [Output Result File] [Platform type]");
		fflush(stdout);
		exit(-1);
	}


	SCLSU_Magma(argv[1],argv[2],argv[3],atoi(argv[4]));


    
    
    
    	return 0;
}




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
		FILE *fp;//evitar warnings
    		if ((fp=fopen(filename,"wb"))!=NULL)
    		{
        		fseek(fp,0L,SEEK_SET);
	    		fwrite(imageSI,1,(lines*samples*bands * sizeof(short int)),fp); free(imageSI);
	    		fclose(fp);
		}
	    	return 0;

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

		FILE *fp;//evitar warnings
    		if ((fp=fopen(filename,"wb"))!=NULL)
    		{
        		fseek(fp,0L,SEEK_SET);
	    		fwrite(imageF,1,(lines*samples*bands * sizeof(float)),fp); free(imageF); 
	    		fclose(fp);	
		}
		return 0;
    
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
		FILE *fp;//evitar warnings
    		if ((fp=fopen(filename,"wb"))!=NULL)
    		{
        		fseek(fp,0L,SEEK_SET);
	    		fwrite(imageD,1,(lines*samples*bands * sizeof(double)),fp); free(imageD); 
	    		fclose(fp);	
    		}
		return 0;
	}

    /*FILE *fp;//evitar warnings
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
    }*/

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

