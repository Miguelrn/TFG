#ifndef GENE_H
#define GENE_H

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



#endif
