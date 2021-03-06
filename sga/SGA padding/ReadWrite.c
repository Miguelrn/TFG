/*
 * ReadWrite.c
 *
 *  Created on: 2013
 *      Author: gabrielma
 */

#include "ReadWrite.h"


// It gets a char* in and return an empty char*
void cleanString(char *cadena, char *out){

	int i,j;
    	for( i = j = 0; cadena[i] != 0; i+=1){
        	if(isalnum(cadena[i])||cadena[i]=='{'||cadena[i]=='.'||cadena[i]==','){
	        	out[j]=cadena[i];
            		j++;
        	}
    	}
    	for( i = j; out[i] != 0; i+=1)
        	out[j]=0;
}

void readHeader(char* filename, int *Samples, int *Lines, int *numBands, int *dataType){

	FILE *fp;
	char line[200];
	char *ptr;
	int i=0;
	

	if(strstr(filename, ".hdr")==NULL){
		printf("ERROR: El fichero %s no contiene el formato adecuado.Debe tener extension hdr\n", filename);
		system("PAUSE");
		exit(1);
	}

	if ((fp=fopen(filename,"r"))==NULL){
		printf("ERROR %d. No se ha podido abrir el fichero .hdr de la imagen: %s \n", filename);
		system("PAUSE");
		exit(1);
	}
	else{
		fseek(fp,0L,SEEK_SET);

		while(fgets(line, 200, fp)!='\0'){

			//printf(" %s\n", line);
			if(strstr(line, "samples")!=NULL){
				ptr=strrchr(line, ' ');
				ptr= ptr+1;
				*Samples=atoi(ptr);
			}
			if(strstr(line, "lines")!=NULL){
				ptr=strrchr(line, ' ');
				ptr= ptr+1;
				*Lines=atoi(ptr);
			}
			if(strstr(line, "bands")!=NULL){
				ptr=strrchr(line, ' ');
				ptr= ptr+1;
				*numBands=atoi(ptr);
			}
			if(strstr(line, "data type")!=NULL){
				ptr=strrchr(line, ' ');
				ptr= ptr+1;
				*dataType=atoi(ptr);
			}

		}//while
		fclose(fp);
	}//else 
}


//load the image "filename" and do a cast over the original datatype into float datatype. Thus, it can operate with data inside the image.
void Load_Image(char* filename, float *imageVector, int lines_samples, int sampleslinesPAD, int numBands, int bandsPAD, int dataType){

	FILE *fp;
    	short int *tipo_short_int;
    	double *tipo_double;
    	float *tipo_float, value;
    	int i,j;
   
    	// open file "filename" only read
    	if ((fp=fopen(filename,"r"))==NULL){
    		printf("file not found");
        	exit(1);
    	}
    	else{
        	fseek(fp,0L,SEEK_SET);
        	//describe datatype inside the image and cast the datatype into float. Thus, the result image will have float datatype inside.
        	switch(dataType)
        	{
	    		//short int datatype 
            		case 2:
                		tipo_short_int = (short int *) malloc (lines_samples*numBands * sizeof(short int));
                		fread(tipo_short_int,1,(sizeof(short int)*lines_samples*numBands),fp);
                		//Convert image data datatype to float
				for(i=0; i< lines_samples; i+=1){
                			for(j=0; j< numBands; j+=1){
						//if(j<bandsPAD){
							value=(float)tipo_short_int[i+j*lines_samples];
							if(value>0)
                    						imageVector[j+i*bandsPAD]=value;//imageVector[i+j*sampleslinesPAD]=value;
							else
								imageVector[j+i*bandsPAD]=0.0;//imageVector[i+j*sampleslinesPAD]=0.0;
						//}
					}
				}
		                free(tipo_short_int);
                		break;
   	    		//float datatype
            		case 4:
				tipo_float = (float *) malloc (lines_samples*numBands * sizeof(float));
                		fread(tipo_float,1,(sizeof(float)*lines_samples*numBands),fp);
				for(i=0; i< lines_samples; i+=1){
                			for(j=0; j< numBands; j+=1){
						//if(j<bandsPAD){
							value=tipo_float[i+j*lines_samples];
							if(value>0)
								imageVector[j+i*bandsPAD]=value;//imageVector[i+j*sampleslinesPAD]=value;
							else
								imageVector[j+i*bandsPAD]=0.0;//imageVector[i+j*sampleslinesPAD]=0.0;
						//}
					}
				}
                		free(tipo_float);
               			break;
			//double datatype
            		case 5:
				tipo_double = (double *) malloc (lines_samples*numBands * sizeof(double));
                		fread(tipo_double,1,(sizeof(double)*lines_samples*numBands),fp);
				for(i=0; i< lines_samples; i+=1){
                			for(j=0; j< numBands; j+=1){
						//if(j<bandsPAD){
							value=(float)tipo_double[i+j*lines_samples];
							if(value>0)
                						imageVector[j+i*bandsPAD]=value;//imageVector[i+j*sampleslinesPAD]=value;
							else
								imageVector[j+i*bandsPAD]=0.0;//imageVector[i+j*sampleslinesPAD]=0.0;
						//}					
					}
				}
                		free (tipo_double);
                		break;
        	}
    	}
    	//close the file 
    	fclose(fp);
}


/*
  Write the image "imagen" into a new image file "resultado_filename" with number of samples "num_samples", number of lines "num_lines" 
  and number of bands "num_bands".
  This method does not write the header file .hdr, only the image file.
*/
void writeResult( float *imagen, const char* resultado_filename, int num_samples, int num_lines, int num_bands){
    FILE *fp;
    int i, j, np=num_samples*num_lines;
    double* imagent = (double*)malloc(num_bands*np*sizeof(double) ); 

    //open file "resultado_filename"
    if ((fp=fopen(resultado_filename,"wb"))!=NULL)
    {
        fseek(fp,0L,SEEK_SET);
	//allocating memory for the image
	//	short int *img=(short int*)malloc(num_lines*num_samples*num_bands*sizeof(short int));
	//	get info from the image "imagen" and write it into the new image "img"
	//	for(i=0; i< (num_lines*num_samples*num_bands);i++){
	//	  	img[i]=(short int)(imagen[i];
	// 		printf("%d\n", img[i]);
	//	  	fflush(stdout);
 	//	}
	//write the image
	for ( i = 0; i < np*num_bands; i++)
		imagent[i] = (double)imagen[i];				
     	fwrite(imagent,1,(num_lines * num_samples * num_bands * sizeof(double)),fp);
    }
    //close the file
    fclose(fp);
}

// Write the header into ".hdr" file
void writeHeader(const char* outHeader, int samples, int lines, int bands){
    // open the file
    FILE *fp=fopen(outHeader,"w+");
    fseek(fp,0L,SEEK_SET);
    fprintf(fp,"ENVI\ndescription = {\nExported from MATLAB}\n");
    fprintf(fp,"samples = %d", samples);
    fprintf(fp,"\nlines   = %d", lines);
    fprintf(fp,"\nbands   = %d", bands);
    fprintf(fp,"\ndata type = 5");
    fprintf(fp,"\ninterleave = bsq");
    fclose(fp);
}



