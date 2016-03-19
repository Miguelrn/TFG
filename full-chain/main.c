#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <CL/cl.h>
#include <math.h>

//#include "gene.h"
#include "sga.h"
//#include "lsu.h"

#define MAXCAD 100

int main(int argc, char **argv) {


	//Variables para calcular el tiempo
	double t_0,t0, t1, treadImage;

	float *imagen;
	char *tipo = (char*)malloc(MAXCAD*sizeof(char));
	char *imagenhdr = (char*)malloc(MAXCAD*sizeof(char));
	char *imagenbsq = (char*)malloc(MAXCAD*sizeof(char));

	int lines, samples, bands, datatype, i,j;
	int endmember = 19, error, deviceSelected;
	size_t localSize;
	pos *solucion;

	//Tener menos de 4 argumentos es incorrecto
	if (argc < 4) {
		fprintf(stderr, "Uso incorrecto de los parametros ./exe 'ruta imagen' 'local size' 'deviceSelected (0|1|2)'\n");
		exit(1);
	}

	localSize = atoi(argv[2]);
	deviceSelected = atoi(argv[3]);

	/*Load Imagen*/
	t_0 = get_time();
	strcpy(imagenhdr,argv[1]);
	strcat(imagenhdr, ".hdr");
	readHeader(imagenhdr, &samples, &lines, &bands, &datatype);
    	printf("Lines = %d  Samples = %d  Nbands = %d  Data_type = %d\n", lines, samples, bands, datatype);
	imagen = (float*) malloc(samples*lines*bands*sizeof(float));

	strcpy(imagenbsq,argv[1]);
	strcat(imagenbsq, ".bsq");
	Load_Image(imagenbsq, imagen, samples, lines, bands, datatype);


	/*GENE*/
printf("ojo hasta que no este gene completo se calculara 19 endmembers!!\n");
	
	/*SGA*/
	float *endmember_bandas = (float*) calloc(bands*endmember, sizeof(float));
	t0 = get_time();
	treadImage=t0-t_0;
	solucion = sga_gpu(imagen, endmember, samples, lines, bands, deviceSelected, endmember_bandas, treadImage, localSize);
	t1 = get_time();

	
	/*LSU*/




	/*strcpy(imagenbsq,argv[1]);
	strcat(imagenbsq, "SGAResult.bsq");
	writeResult(endmember_bandas, imagenbsq, endmember, 1, bands);
	printf("File with endmembers saved at: %s\n",imagenbsq);*/

	for(i = 0; i < endmember; i++){
		printf("%d: %d - %d\n",i+1,solucion[i].filas, solucion[i].columnas);
	}

	free(imagen);
	free(tipo);
	free(imagenhdr);
	free(imagenbsq);
	free(endmember_bandas);
	
	return 0;

}




