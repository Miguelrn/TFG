#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "funciones.h"

#define DEBUG 0

void creaMatriz(float *endmember, int i , int j, float *imagen, int n, int muestras, int lineas, float *jointpoint);
float calculaVolumen(float *jointpoint, int n);


float calculaVolumen(float *jointpoint, int n){
	//calculamos el factorial de n
	int aux = n,factorial = 1,i,j,k;
	float ratio;
	for(i = n-1; i > 0; i--){//factorial de n, en la funcion llega como n+1!!
		factorial *= i;
	}
	

	//calculamos el determinante de la matriz jointpoint haciendo una matriz triangular superior(determinante es la diagonal principal)


    for(i = 0; i < n; i++){

        for(j = 0; j < n; j++){

            if(j>i){

                ratio = jointpoint[j*n+i]/jointpoint[i*n+i];
				/*if(ratio == 1.0 && j < (n-1)){// si el ratio es igual a uno debo hacer swap
					float aux[n];
					for(k = 0; k < n; k++){
						aux[k] = jointpoint[j*n+k];
					}
					for(k = 0; k < n; k++){
						jointpoint[j*n+k] = jointpoint[(j+1)*n+k];
					}
					for(k = 0; k < n; k++){
						jointpoint[(j+1)*n+k] = aux[k];
					}
					ratio = jointpoint[j*n+i]/jointpoint[i*n+i];
				
				}*/
	            for(k = 0; k < n; k++){

	                jointpoint[j*n+k] -= ratio * jointpoint[i*n+k];//matrix j k

	            }
            }

        }
	}
	float determinante = 1;
	for(i = 0; i < n; i++){
		determinante *= jointpoint[i*n+i];	
	}
	

	
	/*if(global && (n-1) == 2){
		global = 0;
		printf("%f / %d  \n\n",determinante,factorial);

		for(i = 0; i < n; i++){
			for(j = 0; j< n; j++){
				printf("%.2f ",jointpoint[i*n+j]);
			}printf("\n");
		}
	
	}*/
	
	//calculamos el valor absoluto y lo dividimos por el factorial
	if(determinante < 0.0)
		return ((-determinante) / factorial);
	else 
		return determinante / factorial;

}

void creaMatriz(float *endmember, int i , int j, float *imagen, int n, int muestras, int lineas, float *jointpoint){
	int x,z;
	//float *jointpoint = (float*) malloc ((n*n+n+(n+1))*sizeof(float));
	//la funcion es generica, vale tanto para n = 1 como para n > 1 !!

	/*if(n == 1 && i == 0 && j == 0 && DEBUG){//solo para mostrar una iteracion de todas las que hace
		printf("El valor del endmember es: \n");
		for(x = 0; x < n*n; x++){
			printf("%f - ", endmember[x]);
		}printf("\n");
		printf("El valor del imagen es es: \n");
		for(x = 0; x < n; x++){
			printf("%f - ", imagen[i + j*muestras + muestras*lineas*x]);
		}printf("\n");
	}*/


	//primero ponemos unos en la primera linea de la matriz, tantos unos como n valga
	for(x = 0; x < n+1; x++){
		jointpoint[x] = 1;
	}


	//copiamos la matriz endmember
	for(x = 0; x < n+1; x++){//columnas             
		for(z = 0; z < n; z++){//filas
			if(x < n){//n primeros caracteres van a ser unos la siguiente linea esta formada por la matriz endmember y la creada a partir de la imagen i , j (else)
				jointpoint[(n+1) + x + z*(n+1)] = endmember[z + x*n];//antes estaba alverres
			}
			else{//ponemos en la posicion n el vector de la imagen apartir de las posiciones i y j
				jointpoint[(n+1) + x + z*(n+1)] = imagen[i + j*muestras + muestras*lineas*z];
			}
		}
	}

	/*if(n == 2 && i == 0 && j == 0 && DEBUG){
		printf("El valor de la matriz final es: \n");
		for(x = 0; x < (n+1)+n*n+n; x++){
			printf("%f - ", jointpoint[x]);
		}printf("\n");
	}*/
}

pos *sga(float *imagen, int num_endmembers, int muestras , int lineas, int bandas){
	//solucion = (pos*) malloc(num_endmembers*sizeof(pos));
	int n = 1, i = 0, j = 0, primeraVuelta = 1,x;
	float volumen = 0.0;
	srand(time(NULL));
	float *endmember;
	if(DEBUG) printf("muestras: %d, filas: %d, bandas %d\n", muestras, lineas, bandas);

	pos *endmember_index = (pos*) malloc((num_endmembers)*sizeof(pos));//vamos a tener 19 puntos de endmembers
	endmember_index[0].filas = rand() % lineas;//221
 	endmember_index[0].columnas = rand() % muestras;//325
	if(DEBUG) printf("la posicion al azar es: %d - %d\n",endmember_index[0].columnas,endmember_index[0].filas);


	while(n < num_endmembers){
		endmember = (float*) malloc(num_endmembers*num_endmembers*sizeof(float));//como maximo va tener p * p elementos, n x n en cada vuelta y se resetea

		for(i = 0; i < n; i++){//para n
			for(j = 0; j < n; j++){//para las bandas
				endmember[i*(n)+j] = imagen[endmember_index[i].columnas + endmember_index[i].filas*muestras + muestras*lineas*j];//flipa pepinillos !
				//printf("%f = ",endmember[i*(n)+j]);
				//printf("-(%d) [%d][%d]- \n",endmember_index[i].columnas + endmember_index[i].filas*muestras + muestras*lineas*j,endmember_index[i].columnas,endmember_index[i].filas);
			}	
		}//printf("--\n");
		////////----------------------///////
		pos *newendmember_index = (pos*) malloc(1*sizeof(pos));
		float maxVolumen = 0.0;

		for(i = 0; i < muestras; i++){
			for(j = 0; j < lineas; j++){
				//le pasamos el endmember y los puntos i j que vamos a usar, devuelve el array hasta n de ese punto
				float *jointpoint = (float*) malloc ((n*n+n+(n+1))*sizeof(float));;
				creaMatriz(endmember, i , j, imagen, n, muestras, lineas, jointpoint);//esta funcion hace jointpoint, transpone la matriz y hace matrixjoinpoint(unos en la primera fila)
	
				volumen = calculaVolumen(jointpoint, n+1);//se calcula sobre una matriz (n+1)*(n+1)
				//if(n==1 && global && i == 349 && j == 296){printf("%f  \n",volumen); }
				if(volumen > maxVolumen){
					maxVolumen = volumen;
					//if(DEBUG)printf("Volumen(%d)[%d][%d]: %f\n",n,i,j,volumen);
					newendmember_index[0].columnas = i;
					newendmember_index[0].filas = j;
				}
				free(jointpoint);
			}
		}		
		endmember_index[n].columnas = newendmember_index[0].columnas;
		endmember_index[n].filas = newendmember_index[0].filas;
		n++;//printf("Valor del maxVolumen: %f\n",maxVolumen);//<- 18 y 19 no caben en float??????????
	
		if(primeraVuelta){//endmeber_index va tener valores pero creo que no se van a usar, si falla mirar aqui
			n = 1;
			endmember_index[0].filas = newendmember_index[0].filas;
			endmember_index[0].columnas = newendmember_index[0].columnas;
			primeraVuelta--;
		}		
		//printf("\n\n");
		free(newendmember_index);
		free(endmember);
	}

	return endmember_index;

}


float *lectura_archivo(char *ruta, int *muestras, int *lineas, int *bandas, char *tipo){
	FILE *fd;
	char *extension = ".hdr";
	char aux[50];//para copiar la ruta original
	char linea[100];
	strcpy(aux, ruta);
	char *token;
	int datatype;
	size_t tam;
	int i, j ,k;
	float *imagen;
	//char ***imagenAux;

//------------------------------------------------------//
//               BASADO FREADENVI.M                     //
//------------------------------------------------------//
	fd = fopen(strcat(ruta, extension),"r");
	if(fd == NULL){
		printf("Error no se encuentra el archivo .hdr %s\n",ruta);
		exit(1);
	}

	//procesamos el archivo cabecera
	while(feof(fd) == 0){
		fgets(linea, 100, fd);
		if(DEBUG) printf("%s", linea);

		token = strtok(linea, "=");
		//if(DEBUG) printf("%s\n",token);


		if (strcmp(token,"samples ") == 0){
			token = strtok(NULL, "=");
			if(DEBUG) printf("%s.\n",token);
			*muestras = atoi(token);
			if(DEBUG) printf("%d\n", *muestras);

		}else if((strcmp(token,"lines   ") == 0) || (strcmp(token,"lines ") == 0)){//hay algunos .hdr que tienen espacios...
			token = strtok(NULL, "=");
			if(DEBUG) printf("%s.\n",token);
			*lineas = atoi(token);
			//if(DEBUG) printf("%d\n", lineas);

		}else if((strcmp(token,"bands   ") == 0) || (strcmp(token,"bands ") == 0)){
			token = strtok(NULL, "=");
			//if(DEBUG) printf("%s.\n",token);
			*bandas = atoi(token);
			//if(DEBUG) printf("%d\n", bandas);

		}else if((strcmp(token,"data type ") == 0) || (strcmp(token,"data type ") == 0)){
			token = strtok(NULL, "=");
			if(DEBUG) printf("%s.\n",token);
			datatype = atoi(token);
			if(DEBUG) printf("%d\n", datatype);

			switch (datatype) {
				case 1:  tipo="bit8";    break;//char
				case 2:  tipo="int16";   break;//short int
				case 3:  tipo="int32";   break;//int
				case 4:  tipo="float32"; break;//float
				case 5:  tipo="float64"; break;//long float ¿? <----- no deja
				case 12: tipo="uint16";  break;//unsigned short ??
				case 13: tipo="uint32";  break;//unsigned int 
				case 14: tipo="int64";   break;//long int ¿?
				case 15: tipo="uint64";  break;//unsigned long int ¿?
                default: tipo="unknown"; break;
			}
		}	
	}
	fclose(fd);
	printf("Opening %d Cols x %d Lines x %d bands\n", *muestras, *lineas, *bandas);
	printf("of type %s image...\n", tipo);
	//------------- Archivo .bsq ----------------------//
	extension = ".bsq";

	fd = fopen(strcat(aux, extension),"r");
	if(fd == NULL){
		printf("Error no se encuentra el archivo .bsq %s\n",ruta);
		exit(1);
	}

	//imagen = (unsigned char*) malloc((*muestras) * (*lineas) * (*bandas) * sizeof(unsigned char));

	//tam = fread(imagen, sizeof(unsigned char), (*muestras) * (*lineas) * (*bandas), fd);




	imagen = (float*)malloc((*muestras) * (*lineas) * (*bandas) * sizeof(float));
	char 				*imagenAux_char;
	short int 			*imagenAux_short_int;
	int 				*imagenAux_int;
	double	 			*imagenAux_double;
	unsigned short int	*imagenAux_uShort;
	unsigned int		*imagenAux_uInt;
	long int			*imagenAux_longInt;
	unsigned long int	*imagenAux_uLongInt;

	switch(datatype){
		case 1: //byte8 -> char
				imagenAux_char = (char*) malloc ((*muestras) * (*lineas) * (*bandas) * sizeof(char));
				tam = fread(imagenAux_char, sizeof(char), (*muestras) * (*lineas) * (*bandas), fd);
				for(i = 0; i < (*muestras) * (*lineas) * (*bandas); i++) imagen[i] = (float)imagenAux_char[i];//otra forma de hacerlo?????????????
        		if(DEBUG) for(i = 0; i < *muestras; i++) printf("%f - ",imagen[i]);
        		free(imagenAux_char);
				break;

				break;
		case 2: //int16 -> short
				imagenAux_short_int = (short int*) malloc ((*muestras) * (*lineas) * (*bandas) * sizeof(short int));
				tam = fread(imagenAux_short_int, sizeof(short int), (*muestras) * (*lineas) * (*bandas), fd);
				for(i = 0; i < (*muestras) * (*lineas) * (*bandas); i++) imagen[i] = (float)imagenAux_short_int[i];
        		if(DEBUG) for(i = 0; i < *muestras; i++) printf("%f - ",imagen[i]);//cuprite en matlab empieza con 726,726,662,663,694,694,726
        		free(imagenAux_short_int);
				break;
		case 3: //int32 -> int
				imagenAux_int = (int*) malloc ((*muestras) * (*lineas) * (*bandas) * sizeof(int));
				tam = fread(imagenAux_int, sizeof(int), (*muestras) * (*lineas) * (*bandas), fd);
				for(i = 0; i < (*muestras) * (*lineas) * (*bandas); i++) imagen[i] = (float)imagenAux_int[i];
        		if(DEBUG) for(i = 0; i < *muestras; i++) printf("%f - ",imagen[i]);
        		free(imagenAux_int);
				break;
		case 4: //float32 -> float
				tam = fread(imagen, sizeof(float), (*muestras) * (*lineas) * (*bandas), fd);
        		if(DEBUG) for(i = 0; i < *muestras; i++) printf("%f - ",imagen[i]);
				break;
		case 5: //float64 -> long float
				imagenAux_double = (double*) malloc ((*muestras) * (*lineas) * (*bandas) * sizeof(double));
				tam = fread(imagenAux_double, sizeof(double), (*muestras) * (*lineas) * (*bandas), fd);
				for(i = 0; i < (*muestras) * (*lineas) * (*bandas); i++) imagen[i] = (float)imagenAux_double[i];
        		//if(DEBUG) for(i = 0; i < *muestras; i++) printf("%f - ",imagen[i]);
        		free(imagenAux_double);
				break;
		case 12: 
				imagenAux_uShort = (unsigned short int*) malloc ((*muestras) * (*lineas) * (*bandas) * sizeof(unsigned short int));
				tam = fread(imagenAux_uShort, sizeof(unsigned short int), (*muestras) * (*lineas) * (*bandas), fd);
				for(i = 0; i < (*muestras) * (*lineas) * (*bandas); i++) imagen[i] = (float)imagenAux_uShort[i];
        		//if(DEBUG) for(i = 0; i < *muestras; i++) printf("%f - ",imagen[i]);
        		free(imagenAux_uShort);
				break;
		case 13: 
				imagenAux_uInt = (unsigned int*) malloc ((*muestras) * (*lineas) * (*bandas) * sizeof(unsigned int));
				tam = fread(imagenAux_uInt, sizeof(unsigned int), (*muestras) * (*lineas) * (*bandas), fd);
				for(i = 0; i < (*muestras) * (*lineas) * (*bandas); i++) imagen[i] = (float)imagenAux_uInt[i];
        		//if(DEBUG) for(i = 0; i < *muestras; i++) printf("%f - ",imagen[i]);
        		free(imagenAux_uInt);
				break;
		case 14: 
				imagenAux_longInt = (long int*) malloc ((*muestras) * (*lineas) * (*bandas) * sizeof(long int));
				tam = fread(imagenAux_longInt, sizeof(long int), (*muestras) * (*lineas) * (*bandas), fd);
				for(i = 0; i < (*muestras) * (*lineas) * (*bandas); i++) imagen[i] = (float)imagenAux_longInt[i];
        		//if(DEBUG) for(i = 0; i < *muestras; i++) printf("%f - ",imagen[i]);
        		free(imagenAux_longInt);
				break;
		case 15: 
				imagenAux_uLongInt = (unsigned long int*) malloc ((*muestras) * (*lineas) * (*bandas) * sizeof(unsigned long int));
				tam = fread(imagenAux_uLongInt, sizeof(unsigned long int), (*muestras) * (*lineas) * (*bandas), fd);
				for(i = 0; i < (*muestras) * (*lineas) * (*bandas); i++) imagen[i] = (float)imagenAux_uLongInt[i];
        		//if(DEBUG) for(i = 0; i < *muestras; i++) printf("%f - ",imagen[i]);
        		free(imagenAux_uLongInt);
				break;

	}         

	if(tam == 0 || tam != ((*muestras) * (*lineas) * (*bandas))){
		printf("Error en la lectura del archivo, tam leido: \n");
		exit(1);
	}
	


	//reshape <-- !!

	/*for(i = 0; i < muestras; i++){
		for(j = 0; j < lineas; j++){
			for(k = 0; k < bandas; k++){

			}
		}
	}*/
	/*for(i = muestras*lineas; i < muestras*lineas+10; i++)
		printf("%c -",imagen[i]);*/

	fclose(fd);

	return(imagen);
}


double get_time(){
	static struct timeval 	tv0;
	double time_, time;

	gettimeofday(&tv0,(struct timezone*)0);
	time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
	time = time_/1000000;
	return(time);
}
