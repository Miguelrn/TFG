#ifndef GENE_H
#define GENE_H

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>

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


void gene_magma(double *image, int samples, int lines, int bands, int Nmax, int P_FA);

void exitOnFail3(cl_int status, const char* message);

#endif
