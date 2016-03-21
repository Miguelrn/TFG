/*
 * ReadWrite.h
 *
 *  Created on: 04/12/2013
 *      Author: gabrielma
 */



#ifndef READWRITE_H_
#define READWRITE_H_

#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
//#include <stddef.h>
#include <sys/time.h>
#include <ctype.h>


void cleanString(char *cadena, char *out);
void readHeader(char* filename, int *Samples, int *Lines, int *numBands, int *dataType);
void Load_Image(char* filename, float *imageVector, int Samples, int Lines, int numBands, int dataType);
void writeResult(double *imagen, const char* resultado_filename, int num_samples, int num_lines, int num_bands);
void writeHeader(const char* outHeader, int samples, int lines, int bands);
double get_time();
//void exitOnFail(cl_int status, const char* message);

#endif /* READWRITE_H_ */

