#ifndef GENE_H
#define GENE_H

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "init_platform.h" 
#include "lsu.h"
#include <gsl/gsl_sf_gamma.h>

//ClMagma Library
#include <magma.h>
#include <magma_lapack.h>


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

void UtxU(double *umatrix, double *mul_umatrix, int iter,int num_bands);
int GaussSeidel_seq (double *matrix, double *inv_matrix, int size);
void CreateExtMatrix (double* src, double* dest, int size, int size_x2);
int ProcessLowerLeftDiag (double *matrix, int size, int size_x2);
int ProcessUpperRightDiag (double *matrix, int size, int size_x2);
void ProcessDiag (double *matrix, int size, int size_x2);
void Uxinv(double *umatrix, double *mul_umatrix_inv, double *umatrix_aux, int iter,int num_bands);
void AnsxUt(double *umatrix_aux, double *umatrix, double *proymatrix, int iter, int num_bands);
void SustractIdentity(double *proymatrix, int num_bands);
double GENE_NP_test(double* theta, int Nmax, int i, double* M, double* y, double* invRsmall);

int gene_magma(double *image, int samples, int lines, int bands, int Nmax, double P_FA, cl_device_id deviceID, double *umatrix_Host, tiempo *gene);
int est_noise(double *image, magmaDouble_ptr image_Device, int linessamples, int bands, magmaDouble_ptr noise_Device, magma_queue_t queue, tiempo *gene);


#endif
