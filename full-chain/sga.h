
#ifndef SGA_H
#define SGA_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <CL/cl.h>
#include <math.h>
#include "ReadWrite.h" 


typedef struct{
	int filas;
	int columnas;
}pos;

pos *sga_gpu(float *imagen, int num_endmembers, int muestras, int lineas, int bandas, int deviceSelected, float *endmember_bandas, size_t localSize);
void exitOnFail(cl_int status, const char* message);

#endif 
