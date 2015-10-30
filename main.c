#include <stdio.h>
#include <stdlib.h>
#include "funciones.h"

#define DEBUG 1



int global = 1;




int main(int argc, char **argv) {


	//Variables para calcular el tiempo
	double t0, t1;

	float *imagen;
	char *tipo;
	int lineas, muestras, bandas, i;
	int endmember;
	pos *solucion;

	//Tener menos de 3 argumentos es incorrecto
	if (argc < 2) {
		fprintf(stderr, "Uso incorrecto de los parametros ./exe 'ruta imagen' 'numero de Endmemebers'\n");
		exit(1);
	}

	endmember = atoi(argv[2]);
	imagen = lectura_archivo(argv[1], &muestras, &lineas, &bandas, tipo);
	//for(i = 0; i < muestras; i++) printf("%f - ",imagen[i]);


	t0 = get_time();
	solucion = sga(imagen, endmember, muestras, lineas, bandas);
	t1 = get_time();

	if(DEBUG) printf("Ha tardado en ejecutarse: %f \n", t1-t0);

	for(i = 0; i < endmember; i++ ){
		printf("%d: %d - %d\n",i+1,solucion[i].columnas,solucion[i].filas);//cuprite -> (298,194)(39,208)(298,206)(297,193)(63,162)
	}


	return 0;

}
