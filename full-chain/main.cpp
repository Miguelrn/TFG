#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <CL/cl.h>
#include <math.h>


#include "ReadWrite.h"
#include "init_platform.h"
#include "gene.h"
#include "sga.h"
#include "lsu.h"

#define MAXCAD 100

int main(int argc, char **argv) {


	//Variables para calcular el tiempo
	double t_0,t0, t1, treadImage;

	double *imagen_h;
	char *tipo = (char*)malloc(MAXCAD*sizeof(char));
	char *imagenhdr = (char*)malloc(MAXCAD*sizeof(char));
	char *imagenbsq = (char*)malloc(MAXCAD*sizeof(char));
	char endmember_file[MAXCAD];
	double *endmember_bandas_h;
	double *umatrix_h;
	double *abundancias_h;

	int lines, samples, bands, datatype, i,j;
	int linesEnd, samplesEnd, bandsEnd, datatypeEnd;//se podria comprobar que lines y linesEnd son iguales... etc

	int endmember = 19, error, deviceSelected, librarySelected = 0, maxEndmembers;
	float probFail;
	size_t localSize;
	pos *solucion;

	cl_context context;
	cl_command_queue command_queue;
	cl_device_id deviceID;

	//Tener menos de 8 argumentos es incorrecto
	if (argc != 8) {
		fprintf(stderr, "Incorrect parameters: ./exe 'ruta imagen' 'Max Endmembers' 'Fail probability' 'local size' 'deviceSelected (0|1|2)' 'ViennaCL = 1, CLMagma = 2' 'a|b|c|d|e'\n");
		fprintf(stderr,"a) GENE\nb) GENE + SCLSU\nc) SGA\nd) SCLSU\ne) GENE + SGA + SCLSU\n");
		exit(1);
	}

	maxEndmembers = atoi(argv[2]);
	probFail = atof(argv[3]);
	localSize = atoi(argv[4]);
	deviceSelected = atoi(argv[5]);
	librarySelected = atoi(argv[6]);
	if(librarySelected != 1 && librarySelected != 2){
		printf("Library selected is not found, use 1 for ViennaCl or 2 for ClMagma\n");
		exit(-1);
	}

	printf("-----------------------------------------------------------------------\n");
	printf("                    Imagen: %s\n",argv[1]);
	printf("-----------------------------------------------------------------------\n");
	/*Load Imagen*/
	t0 = get_time();
	strcpy(imagenhdr,argv[1]);
	strcat(imagenhdr, ".hdr");
	readHeader(imagenhdr, &samples, &lines, &bands, &datatype);
    	printf("Lines = %d  Samples = %d  Nbands = %d  Data_type = %d\n", lines, samples, bands, datatype);
	MALLOC_HOST(imagen_h, double, samples*lines*bands)


	strcpy(imagenbsq, argv[1]);
	strcat(imagenbsq, ".bsq");
	Load_Image(imagenbsq, imagen_h, samples, lines, bands, datatype);
	t1 = get_time();
	printf("Tiempo de lectura de la Imagen: %f\n",t1-t0);

	t0 = get_time();
	init_OpenCl(&context, &command_queue, deviceSelected, &deviceID);
	//init magma
	//init Viennacl ?
	t1 = get_time();

	switch(argv[7][0]){
		case 'a':/* GENE */
			MALLOC_HOST(umatrix_h, double, maxEndmembers*maxEndmembers)
			gene_magma(imagen_h, samples, lines, bands, maxEndmembers, probFail, command_queue, context, deviceID, umatrix_h);
			magma_free_cpu(umatrix_h);
			break;

		case 'b':/* GENE + SCLSU */
			MALLOC_HOST(umatrix_h, double, maxEndmembers*maxEndmembers)
			endmember = gene_magma(imagen_h, samples, lines, bands, maxEndmembers, probFail, command_queue, context, deviceID, umatrix_h);

			MALLOC_HOST(abundancias_h, double, endmember*lines*samples)
			lsu_gpu_m(imagen_h, umatrix_h, deviceID, maxEndmembers, endmember, lines, samples, argv[1], abundancias_h);
			magma_free_cpu(abundancias_h);
			magma_free_cpu(umatrix_h);
			break;

		case 'c':/* SGA */
			printf("Please indicate how many endmembers need to be found:\n");
			scanf ("%d",&endmember);
			fflush(stdin);
			MALLOC_HOST(endmember_bandas_h, double, bands*endmember)
			t0 = get_time();
			solucion = sga_gpu(imagen_h, endmember, samples, lines, bands, endmember_bandas_h, localSize, context, command_queue);
			t1 = get_time();
			for(i = 0; i < endmember; i++){
				printf("%2d: %d - %d\n",i+1,solucion[i].filas, solucion[i].columnas);
			}
			break;

		case 'd':/* SCLSU */
			printf("Please include the path where the endmember file is stored:\n");
			std::cin >> endmember_file;
			
			strcpy(imagenhdr,endmember_file);
			strcat(imagenhdr, ".hdr");
			readHeader(imagenhdr, &samplesEnd, &linesEnd, &bandsEnd, &datatypeEnd);
			MALLOC_HOST(endmember_bandas_h, double, linesEnd*bandsEnd)
			//printf("Lines = %d  Samples = %d  Nbands = %d  Data_type = %d\n", linesEnd, samplesEnd, bandsEnd, datatypeEnd);

			strcpy(imagenbsq, endmember_file);
			strcat(imagenbsq, ".bsq");
			Load_Image(imagenbsq, endmember_bandas_h, samplesEnd, linesEnd, bandsEnd, datatypeEnd);

			t0 = get_time();
			if(librarySelected == 1)//ViennaCl
				lsu_gpu_v(imagen_h, endmember_bandas_h, deviceSelected, bands, endmember, lines, samples, argv[1]);
			else if(librarySelected == 2){//ClMagma
				MALLOC_HOST(abundancias_h, double, linesEnd*lines*samples)
				lsu_gpu_m(imagen_h, endmember_bandas_h, deviceID, bands, endmember, lines, samples, argv[1], abundancias_h);
				magma_free_cpu(abundancias_h);
			}
			t1 = get_time();
			break;

		case 'e':/* GENE + SGA + SCLSU */
			/*GENE*/
			MALLOC_HOST(umatrix_h, double, maxEndmembers*maxEndmembers)
			endmember = gene_magma(imagen_h, samples, lines, bands, maxEndmembers, probFail, command_queue, context, deviceID, umatrix_h);
	
			/*SGA*/
			MALLOC_HOST(endmember_bandas_h, double, bands*endmember)
			t0 = get_time();
			solucion = sga_gpu(imagen_h, endmember, samples, lines, bands, endmember_bandas_h, localSize, context, command_queue);
			t1 = get_time();


			/*LSU*/
			t0 = get_time();
			if(librarySelected == 1)//ViennaCl
				lsu_gpu_v(imagen_h, endmember_bandas_h, deviceSelected, bands, endmember, lines, samples, argv[1]);
			else if(librarySelected == 2){//ClMagma
				MALLOC_HOST(abundancias_h, double, endmember*lines*samples)
				lsu_gpu_m(imagen_h, endmember_bandas_h, deviceID, bands, endmember, lines, samples, argv[1], abundancias_h);
				magma_free_cpu(abundancias_h);
			}
			t1 = get_time();
			magma_free_cpu(umatrix_h);

			break;

		default: printf("case not supported, exiting...\n"); exit(-1);
	}
	





	magma_free_cpu(imagen_h);
	free(tipo);
	free(imagenhdr);
	free(imagenbsq);
	clReleaseCommandQueue(command_queue);
    	clReleaseContext(context);
	
	return 0;

}




