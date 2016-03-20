#ifndef LSU_H
#define LSU_H

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "ReadWrite.h"

#define MAXLINE 200
#define MAXCAD 90

//Viena Opencl Libraries
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/tools/timer.hpp"
#include "viennacl/linalg/lu.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/linalg/sum.hpp"

#ifndef VIENNACL_WITH_OPENCL
	#define VIENNACL_WITH_OPENCL 
#endif

//uBlas Headers
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>


//lsu whit ViennaCl
void lsu_gpu_v(float *imagen, float *endmembers, int DeviceSelected, int bandas, int targets, int lines, int samples, char *filename);

//lsu whit Clmagma
//void lsu_gpu_m(float *imagen, float *endmembers, int DeviceSelected, int bandas, int targets, int lines, int samples, char *filename)


#endif
