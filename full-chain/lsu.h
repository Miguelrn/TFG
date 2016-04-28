#ifndef LSU_H
#define LSU_H

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "ReadWrite.h"


//Vienna Opencl Libraries
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/tools/timer.hpp"
#include "viennacl/linalg/lu.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/linalg/sum.hpp"
//#include "viennacl/linalg/svd.hpp"



//ClMagma Library
#include <magma.h>
#include <magma_lapack.h>


#define MAXLINE 200
#define MAXCAD 90


#ifndef VIENNACL_WITH_OPENCL
	#define VIENNACL_WITH_OPENCL 
#endif

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

//lsu whit ViennaCl
void lsu_gpu_v(double *imagen, double *endmembers, int DeviceSelected, int bandas, int targets, int lines, int samples, char *filename);



//lsu whit Clmagma
void lsu_gpu_m(double *imagen, double *endmembers, cl_device_id deviceID, int bandas, int targets, int lines, int samples, char *filename);
void IF1_Aux(double* IF,double* IF1, double* Aux,int targets);
void UFdiag(double* UF,double* SF,double* IF,int targets,double mu);
void divide_norm(double *X, double* M, double norm, int lines_samples, int bands, int p);
double avg_X_2(double *X, int lines_samples, int num_bands);



#endif
