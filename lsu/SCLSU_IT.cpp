#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define MAXLINE 200
#define MAXCAD 90


//Viena Opencl Libraries
/*#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/tools/timer.hpp"
#include "viennacl/linalg/lu.hpp"

#ifndef VIENNACL_WITH_OPENCL
	#define VIENNACL_WITH_OPENCL 
#endif*/

//g++ -o SCLSU SCLSU_IT.cpp -I/usr/include/viennacl/ -lblas -llapack -lOpenCL -DVIENNACL_WITH_OPENCL


extern "C" int dgemm_(char const *transa, char const *transb, int *m, int *
		n, int *k, double *alpha, double *a, int *lda,
		double *b, int *ldb, double *beta, double *c, int
		*ldc);

extern "C" int dgetrf_(int *m, int *n, double *a, int *
	lda, int *ipiv, int *info);

extern "C" int dgetri_(int *n, double *a, int *lda, int
	*ipiv, double *work, int *lwork, int *info);

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
	if(argc !=  4)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Parameters are not correct.");
		printf("./SCLSU [Image Filename] [Endmembers file] [Output Result File]");
		fflush(stdout);
		exit(-1);
	}

	//READ IMAGE
	char header_filename[MAXCAD];
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

	double *image = (double*)malloc(lines*samples*bands*sizeof(double));
	error = loadImage(argv[1], image, lines, samples, bands, dataType, interleave);
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

	//START CLOCK***************************************
	clock_t start, end;
	start = clock();
	//**************************************************
//	MatrixType I(N_endmembers, N_endmembers);
//	I.fill(0);
//	for(i=0; i<N_endmembers;i++)
//		I.put(i,i,1);
//
//	//INVERSE (E_t * E)⁻¹
//	//E_t
//	MatrixType END_T = END.transpose();//(n_end x bands)
//	//E_t * E
//	MatrixType PRODUCT = END_T * END; //(n_end x n_end)
//	//(E_t * E)⁻¹
//	MatrixType INVERSE = vnl_matrix_inverse<double>(PRODUCT);//(n_end x n_end)
//
//	//[1_t * (E_t *E)⁻¹ *1]⁻¹ == X
//	//ONE
//	MatrixType ONE(N_endmembers,1);
//	ONE.fill(1);
//	//ONET
//	MatrixType ONE_T = ONE.transpose(); (1, N_endmembers)
//	MatrixType X = vnl_matrix_inverse<double>(ONE_T * INVERSE * ONE); (1,1)
//
//	//SCLS = [I - INV*ONE*X*ONET] [INV*ENDT*PIXEL(i,j)] + [INV*ONE*X]
//	MatrixType A = I-(INVERSE*ONE*X*ONE_T);
//	VectorType B = INVERSE*ONE*X.get_row(0);
//
//	ABUNDANCES.set_size(N_endmembers,N_samples*N_lines);
//
//	for(i=0;i<N_samples*N_lines;i++)
//		ABUNDANCES.set_column(i,A*(INVERSE*END_T*PIXELS.get_column(i)) + B);
	int i, j, one = 1;
	double alpha = 1, beta = 0;
	double *Et_E = (double*)malloc(targets*targets*sizeof(double));

	//--------------------------------------------//
/*
	int numero = 2;
	double identidad[numero*numero] = {1,0,0,1};
	double M_A[numero*numero] = {1,2,3,4};
	double M_B[numero*numero] = {1,3,2,4};
	double M_C[numero*numero] = {0,0,0,0};

	//for(i =0; i < numero; i++){ for(j =0; j < numero; j++) printf("%f ", M_A[i*numero+j]); printf("\n");}

	dgemm_("T", "N", &numero, &numero, &numero, &alpha, M_A, &numero, identidad, &numero, &beta, M_C, &numero);
	for(i = 0; i < numero*numero; i++) printf("%f ", M_C[i]); printf("\n");//1,3,2,4

	dgemm_("T", "T", &numero, &numero, &numero, &alpha, identidad, &numero, M_B, &numero, &beta, M_C, &numero);
	for(i = 0; i < numero*numero; i++) printf("%f ", M_C[i]); printf("\n");//1,2,3,4

	dgemm_("T", "N", &numero, &numero, &numero, &alpha, M_A, &numero, M_A, &numero, &beta, M_C, &numero);//ojo ublas usa la traspuesta con column major 
	for(i = 0; i < numero*numero; i++) printf("%f ", M_C[i]); printf("\n");//deberia ser 10 14 14 20 -> 5 11 11 25

	dgemm_("N", "T", &numero, &numero, &numero, &alpha, M_A, &numero, M_A, &numero, &beta, M_C, &numero);
	for(i = 0; i < numero*numero; i++) printf("%f ", M_C[i]); printf("\n");//deberia ser 5 11 11 25 -> 10 14 14 20

	int inf;
	int resultado[numero];
	double trabajo[numero];
	dgetrf_(&numero, &numero, M_B, &numero, resultado, &inf);
	printf("El resultado de dgetrf_: %d \n",inf);
	for(i = 0; i < numero*numero; i++) printf("%f ", M_B[i]); printf("\n");
	for(i = 0; i < numero; i++) printf("%d ", resultado[i]); printf("\n");

	dgetri_(&numero, M_B, &numero, resultado, trabajo, &numero, &inf);
	printf("El resultado de dgetri_: %d \n",inf);
	for(i = 0; i < numero*numero; i++) printf("%f ", M_B[i]); printf("\n");
	for(i = 0; i < numero; i++) printf("%f ", trabajo[i]); printf("\n");
*/
	//--------------------------------------------//
 	/*viennacl::ocl::set_context_device_type(0, viennacl::ocl::gpu_tag());
    std::vector<viennacl::ocl::device> devices = viennacl::ocl::current_context().devices();
   	viennacl::ocl::current_context().switch_device(devices[0]);

	std::vector<cl_device_id> device_id_array;
	//get all available devices
	viennacl::ocl::platform pf;
	std::cout << "Platform info: " << pf.info() << std::endl;
	devices = pf.devices(CL_DEVICE_TYPE_DEFAULT);
	std::cout << devices[0].name() << std::endl;
	std::cout << "Number of devices for custom context: " << devices.size() << std::endl;*/
	//--------------------------------------------//
	/*int filas = 2, columnas = 3;
	double *A1 = (double*)malloc(2*3*sizeof(double)); for (unsigned int i = 0; i < 2*3; i++) A1[i] = i+1;//simulara el vector endmembers (19*188)
	double *iden = (double*) malloc(2*2*sizeof(double));iden[0] = 1; iden[3] = 1;//vamos a mirar la traspuesta
	double *B1 = (double*)malloc(2*2*sizeof(double)); //simulara ser Et_E
	dgemm_("N", "T", &filas, &filas, &columnas, &alpha, A1, &filas, A1, &filas, &beta, B1, &filas);
	for (unsigned int i = 0; i < filas; i++) {
		for (unsigned int j = 0; j < filas; j++){
			printf("%.1f ",B1[i*filas+j]);
		}printf("\n");
	}
	printf("---\n");
	std::vector<std::vector<double> > A1_CPU(filas);//este vector de vectores sera A1 formateado
	std::vector<std::vector<double> > B1_CPU(filas);
    viennacl::matrix<double> A1_GPU(filas, columnas);
    viennacl::matrix<double> B1_GPU(filas, filas);
	for (unsigned int i = 0; i < filas; i++) {
		A1_CPU[i].resize(columnas);
		B1_CPU[i].resize(filas);

		for (unsigned int j = 0; j < columnas; j++){
		    A1_CPU[i][j] = A1[i + j*filas];
			
		}
	}
	for (unsigned int i = 0; i < filas; i++) {//lo copia bien
		for (unsigned int j = 0; j < columnas; j++){
			printf("%.1f ",A1_CPU[i][j]);
		}printf("\n");
	}
	printf("---\n");
	viennacl::copy(A1_CPU, A1_GPU);
    B1_GPU = viennacl::linalg::prod(A1_GPU, viennacl::trans(A1_GPU));

    viennacl::copy(B1_GPU, B1_CPU);
	for (unsigned int i = 0; i < filas; i++) {
		for (unsigned int j = 0; j < filas; j++){
			printf("%.1f ",B1_CPU[i][j]);
		}printf("\n");
	}

ESTO FUNCIONA DEJARLO COMO PLANTILLA PORQUE COMO TENGA QUE PENSARLO DE NUEVO ME MUERO!
*/

 /*   //Host
	std::vector<std::vector<double> > endmembers_CPU(targets);//19
	std::vector<std::vector<double> > EtE_CPU(targets);

    //Device
	viennacl::matrix<double> endmembers_GPU(targets, bandsEnd);//19*188
    viennacl::matrix<double> EtE_GPU(targets, targets);
    viennacl::vector<double> work_GPU(targets);
    viennacl::vector<double> vcl_result(targets);

	for (unsigned int i = 0; i < targets; i++) {
		endmembers_CPU[i].resize(bandsEnd);
		EtE_CPU[i].resize(targets);

		for (unsigned int j = 0; j < bandsEnd; j++){
		    endmembers_CPU[i][j] = endmembers[i + j*targets];
		}
	}
	printf("---\n");
	viennacl::copy(endmembers_CPU, endmembers_GPU);

	EtE_GPU = viennacl::linalg::prod(endmembers_GPU, viennacl::trans(endmembers_GPU));
	

    vcl_result = viennacl::linalg::solve(EtE_GPU, work_GPU, viennacl::linalg::upper_tag());
    vcl_result = viennacl::linalg::solve(EtE_GPU, work_GPU, viennacl::linalg::lower_tag());	
    viennacl::linalg::lu_factorize(EtE_GPU);
    viennacl::linalg::lu_substitute(EtE_GPU, work_GPU);
	

	/*for(unsigned i = 0; i < targets; i++){//solo para comprobar que hace la multiplicacion bien tanto con lapack como con ViennaCL
		for(unsigned j = 0; j < targets; j++){
			if(fabs(EtE_CPU[i][j]-Et_E[i*targets+j]) > 0.01){
				printf("Error \n");
			}
		}
	}*/
	//viennacl::copy(EtE_GPU, EtE_CPU);
//-----------------------------------------------------------------------------------------------------------------------------
printf("//--------------------------------------------//\n");
	int seis = 16, a,b;
	int sizes[seis] = {32,64,128,192,320,640,1088,2112,32,64,128,192,320,640,1088,2112};
	for(a = 0; a < seis; a++){
		int M = sizes[a];
		double *h_A,*h_C;
		h_A = (double*) malloc (M*M*sizeof(double));
		h_C = (double*) malloc (M*M*sizeof(double));
		for(b = 0; b < M*M; b++) h_A[b] = b;
		
		double z0 = clock();
		dgemm_("N", "T", &M, &M, &M, &alpha, h_A, &M, h_A, &M, &beta, h_C, &M);
		z0 = clock() - z0;
		printf("tiempo eth(%d): %f\n", sizes[a], (double)(z0)/CLOCKS_PER_SEC);

		free(h_A);free(h_C);
	}
printf("//--------------------------------------------//\n");
//-----------------------------------------------------------------------------------------------------------------------------

	//------------------------------------------------//
	double h1 = clock();
    	dgemm_("N", "T", &targets, &targets, &bandsEnd, &alpha, endmembers, &targets, endmembers, &targets, &beta, Et_E, &targets);
	double j1 = clock();
	printf("tiempo eth: %f\n", (double)(j1-h1)/CLOCKS_PER_SEC);
	//for(unsigned i = 0; i < targets*targets; i++){printf("%f ",Et_E[i]);}
	int *ipiv = (int*)malloc(targets*sizeof(int));
	int info;
	int lwork = targets;
	double *work = (double*)malloc(lwork*sizeof(double));
	dgetrf_(&targets,&targets,Et_E,&targets,ipiv, &info);
	dgetri_(&targets, Et_E, &targets, ipiv,work, &lwork, &info);

	//for(unsigned i = 0; i < targets*targets; i++){Et_E[i] = 2;}//-----------------------------------------------------------------------------------------------
//-----------------------
//Pruebas en una matriz mas pequeña
/*	srand (time(NULL));
	int filas = 188, columnas = 19;
	double *prueba = (double*)malloc(filas*columnas*sizeof(double));//matriz de 3*6 -> simula el vector endmember
	double *resultado = (double*)malloc(columnas*columnas*sizeof(double));//matriz de 3*3 -> simula el vector Et_E
	for(unsigned i = 0; i < filas*columnas; i++){
		prueba[i] = rand() % 10; printf("%f ",prueba[i]);
	}printf("\n");
	dgemm_("N", "T", &columnas, &columnas, &filas, &alpha, prueba, &columnas, prueba, &columnas, &beta, resultado, &columnas);
	for(unsigned i = 0; i < columnas*columnas; i++) printf("%f ",resultado[i]); printf("\n");
	int *pivote = (int*)malloc(columnas*sizeof(int));
	int ltrabajo = columnas;
	double *trabajo = (double*)malloc(ltrabajo*sizeof(double));
	dgetrf_(&columnas,&columnas,resultado,&columnas,pivote, &info);
	printf("El resultado de la factorizacion es: %d\n",info);
	dgetri_(&columnas, resultado, &columnas, pivote, trabajo, &ltrabajo, &info);
	printf("El resultado de la inversion es: %d\n",info);
	for(unsigned i = 0; i < columnas*columnas; i++) printf("%f ",resultado[i]); printf("\n");
*/    
//----------------------
	double* ONE = (double*)malloc(targets*sizeof(double));
	for(i=0; i<targets; i++) ONE[i] = 1;
	double* AUX = (double*)calloc(targets,sizeof(double));
	double z2 = clock();
/**/	dgemm_("N", "N", &one, &targets, &targets, &alpha, ONE, &one, Et_E, &targets, &beta, AUX, &one);
	double z1 = clock();
	printf("\ntiempo: %f \n",(double)(z1-z2)/CLOCKS_PER_SEC);
	//for(unsigned i = 0; i < targets; i++){printf("%f ",AUX[i]);}
//---------
	double Y = 0;
	for(i=0; i<targets; i++) Y += AUX[i]; printf("Y: %f\n",Y);
	Y = 1 / Y;
	for(i=0; i<targets; i++) AUX[i] = Y;
	//for(unsigned i = 0; i < targets; i++){printf("%f ",AUX[i]);}
//---------
	double* AUX2 = (double*)calloc(targets,sizeof(double));
/**/	dgemm_("N", "N", &one, &targets, &targets, &alpha, AUX, &one, Et_E, &targets, &beta, AUX2, &one);


	double* I = (double*)calloc(targets*targets,sizeof(double));
	for(i=0; i<targets; i++)
		for(j=0; j<targets; j++)
			if(i == j) I[i*targets+j] = 1 - AUX2[j];
			else I[i*targets+j] = -AUX2[j];

	//for(unsigned i = 0; i < targets*targets; i++){printf("%f ",I[i]);}printf("\n");
//---------
	double* A = (double*)calloc(targets*targets,sizeof(double));
/**/	dgemm_("N", "N", &targets, &targets, &targets, &alpha, I, &targets, Et_E, &targets, &beta, A, &targets);

	//for(unsigned i = 0; i < targets*targets; i++){printf("%f ",A[i]);}printf("\n");

	for(unsigned i = 0; i < targets; i++){A[i*targets+i] = 1;}

	double* B = (double*)calloc(targets*bands,sizeof(double));
/**/	dgemm_("N", "N", &targets, &bands, &targets, &alpha, A, &targets, endmembers, &targets, &beta, B, &targets);

	//for(unsigned i = 0; i < targets*bandsEnd; i++){printf("%f ",B[i]);}printf("\n");

/**/	dgemm_("N", "N", &targets, &one, &targets, &alpha, Et_E, &targets, ONE, &targets, &beta, AUX, &targets);
	for(i=0; i<targets; i++) AUX[i] *= Y;
//---------

	double* ABUN = (double*)calloc(lines*samples*targets, sizeof(double));
	double* PIXEL = (double*)calloc(bands, sizeof(double));
	double h = clock();
	for(i=0; i<lines*samples; i++)
	{
		for(j=0; j<bands; j++){ PIXEL[j] = image[j*lines*samples+i]; /*if(i == 0) printf("%f ",PIXEL[j]);*/}
/**/		dgemm_("N", "N", &targets, &one, &bands, &alpha, B, &targets, PIXEL, &bands, &beta, AUX2, &targets);
		if(i == 0) for(j=0; j<targets; j++){ printf("%f ",AUX2[j]);}

		for(j=0; j<targets; j++) ABUN[j*lines*samples + i] = AUX2[j] + AUX[j];
	}
	double z = clock();
	printf("\nBucle gordo %f \n",(double)(z-h)/CLOCKS_PER_SEC);
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

	error = writeResult(ABUN,argv[3],lines,samples, targets, 5, interleave);
	if(error != 0)
	{
		printf("EXECUTION ERROR SCLSU Iterative: Error writing endmembers file: %s.", argv[3]);
		fflush(stdout);
		return error;
	}

	//FREE MEMORY***************************************
	free(Et_E);
	free(ipiv);
	free(work);
	free(ONE);
	free(AUX);
	free(AUX2);
	free(I);
	free(A);
	free(B);
	free(PIXEL);
	free(ABUN);
	free(wavelength);
	free(image);
	free(endmembers);
	free(interleaveE);
	free(interleave);
	free(waveUnit);

	return 0;
}






























