
#ifndef SGA_H
#define SGA_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
#include "ReadWrite.h"
#include "init_platform.h" 


typedef struct{
	int filas;
	int columnas;
}pos;

pos *sga_gpu(double *imagen, int num_endmembers, int muestras, int lineas, int bandas, double *endmember_bandas, size_t localSize, cl_context context, cl_command_queue command_queue, tiempo *sga);


#endif 
