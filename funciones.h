#ifndef FUNCIONES_H
#define FUNCIONES_H

typedef struct{
	int filas;
	int columnas;
}pos;

double get_time();

float *lectura_archivo(char *ruta, int *lineas, int *muestras, int *bandas, char *tipo);
pos *sga(float *imagen, int num_endmembers, int muestras , int lineas, int bandas);



#endif
