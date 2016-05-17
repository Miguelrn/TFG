#ifndef READWRITE_H_
#define READWRITE_H_

#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <ctype.h>
#include <vector>




void cleanString(char *cadena, char *out);
void readHeader(char* filename, int *Samples, int *Lines, int *numBands, int *dataType);
void Load_Image(char* filename, double *imageVector, int Samples, int Lines, int numBands, int dataType);
void writeResult(double *imagen, const char* resultado_filename, int num_samples, int num_lines, int num_bands);
void writeResult( std::vector< std::vector<double> > imagen, const char* resultado_filename, int num_samples, int num_lines, int num_bands);
void writeHeader(const char* outHeader, int samples, int lines, int bands);
double get_time();


#endif /* READWRITE_H_ */

