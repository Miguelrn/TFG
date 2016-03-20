#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <CL/cl.h>
#include <math.h>

#include "ReadWrite.h"
//#include "gene.h"
#include "sga.h"
#include "lsu.h"

#define MAXCAD 100

int main(int argc, char **argv) {


	//Variables para calcular el tiempo
	double t_0,t0, t1, treadImage;

	float *imagen;
	char *tipo = (char*)malloc(MAXCAD*sizeof(char));
	char *imagenhdr = (char*)malloc(MAXCAD*sizeof(char));
	char *imagenbsq = (char*)malloc(MAXCAD*sizeof(char));

	int lines, samples, bands, datatype, i,j;
	int endmember = 19, error, deviceSelected, librarySelected = 0;
	size_t localSize;
	pos *solucion;

	//Tener menos de 4 argumentos es incorrecto
	if (argc < 5) {
		fprintf(stderr, "Uso incorrecto de los parametros ./exe 'ruta imagen' 'local size' 'deviceSelected (0|1|2)' 'ViennaCL = 1, CLMagma = 2'\n");
		exit(1);
	}

	localSize = atoi(argv[2]);
	deviceSelected = atoi(argv[3]);
	librarySelected = atoi(argv[4]);
	if(librarySelected != 1 && librarySelected != 2){
		printf("Library selected is not found, use 1 for ViennaCl or 2 for ClMagma\n");
		exit(-1);
	}


	/*Load Imagen*/
	t0 = get_time();
	strcpy(imagenhdr,argv[1]);
	strcat(imagenhdr, ".hdr");
	readHeader(imagenhdr, &samples, &lines, &bands, &datatype);
    	printf("Lines = %d  Samples = %d  Nbands = %d  Data_type = %d\n", lines, samples, bands, datatype);
	imagen = (float*) malloc(samples*lines*bands*sizeof(float));

	strcpy(imagenbsq,argv[1]);
	strcat(imagenbsq, ".bsq");
	Load_Image(imagenbsq, imagen, samples, lines, bands, datatype);
	t1 = get_time();
	printf("Tiempo de lectura de la Imagen: %f\n",t1-t0);


	/*GENE*/
printf("ojo hasta que no este gene completo se calculara 19 endmembers!!\n");
	
	/*SGA*/
	float *endmember_bandas = (float*) calloc(bands*endmember, sizeof(float));
	t0 = get_time();
	//treadImage=t0-t_0;//no tiene sentido incluir el tiempo de lectura aqui... ponerlo separado
	solucion = sga_gpu(imagen, endmember, samples, lines, bands, deviceSelected, endmember_bandas, localSize);
	t1 = get_time();

	
	/*LSU*/
	t0 = get_time();
	if(librarySelected == 1)//ViennaCl
		lsu_gpu_v(imagen, endmember_bandas, deviceSelected, bands, endmember, lines, samples, argv[1]);
	//else if(librarySelected == 2)//ClMagma
	//	lsu_gpu_m(imagen, endmember_bandas, deviceSelected, bands, endmember, lines, samples, argv[1]);
	t1 = get_time();


	/*strcpy(imagenbsq,argv[1]);
	strcat(imagenbsq, "SGAResult.bsq");
	writeResult(endmember_bandas, imagenbsq, endmember, 1, bands);
	printf("File with endmembers saved at: %s\n",imagenbsq);*/

	/*for(i = 0; i < endmember; i++){
		printf("%d: %d - %d\n",i+1,solucion[i].filas, solucion[i].columnas);
	}*/

	free(imagen);
	free(tipo);
	free(imagenhdr);
	free(imagenbsq);
	free(endmember_bandas);
	
	return 0;

}




