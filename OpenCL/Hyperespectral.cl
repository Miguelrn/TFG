//QUITARTE LOS MALLOC , en opencl no deja... zzzzz

//OJO posiones es un array de una dimension, las posiciones iran colocadas consecutivamente es decir pos[0] = columnas, pos[1] = filas, pos[2] = columnas....

//OJO nunca usar mas de 19 end_members, Opencl no soporta variable array length
//OJO volumen para calcular la imagen Cuprite (350*350) mas grande va fallar


__global float calculaVolumen(float *jointpoint, int n){
	//calculamos el factorial de n
	int i,j,k;
	int factorial = 1;
	float ratio;
	for(i = n-1; i > 0; i--){//factorial de n, en la funcion llega como n+1!!
		factorial *= i;
	}
	
	//calculamos el determinante de la matriz jointpoint haciendo una matriz triangular superior(determinante es la diagonal principal)
    for(i = 0; i < n; i++){

        for(j = 0; j < n; j++){

            if(j>i){

                ratio = jointpoint[j*n+i]/jointpoint[i*n+i];
				
	            for(k = 0; k < n; k++){
	                //jointpoint[j*n+k] = jointpoint[j*n+k] - (ratio * jointpoint[i*n+k]);//matrix j k

	            }
            }

        }
	}
	float determinante = 1;
	for(i = 0; i < n; i++){
		//determinante = determinante * jointpoint[i*n+i];	
	}
	


	//calculamos el valor absoluto y lo dividimos por el factorial
	/*if(determinante < 0.0)
		return ((-determinante) / factorial);
	else 
		return determinante / factorial;//return determinante;*/
	if(determinante < 0.0)
		return ((-determinante) / factorial);
	else 
		return ((determinante) / factorial);
}


void creaMatriz(float *endmember, int i , int j, __global float *imagen, int n, int muestras, int lineas, float *jointpoint){
	int x,z;
	//float *jointpoint = (float*) malloc ((n*n+n+(n+1))*sizeof(float));

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

}


__kernel void endmembers_calculation(__global float *ImageIn, __global int *posiciones, __global float *volumen, const int num_endmembers, const int muestras, const int lineas, const int bandas, __global float *mierda){
	
	const int idx = get_global_id(0); 
	const int jdx = get_global_id(1);
	//const int zdx = get_global_id(2);

	const int MAX_ARRAY_SIZE = num_endmembers*num_endmembers;

	int n = 1, i = 0, j = 0, primeraVuelta = 1,x,k,a;
	//float volumen[35*35];

	float endmember[19*19], ratio;
	float maxVolumen;
	posiciones[0] = 325;
	posiciones[1] = 221;//esto deberia ser random

if((idx*muestras + jdx) < muestras*lineas){//por si acaso se sale...
	

	while(n < num_endmembers){


		for(i = 0; i < n; i++){//para n
			for(j = 0; j < n; j++){//para las bandas
				endmember[i*(n)+j] = ImageIn[posiciones[i*2] + posiciones[i*2+1]*muestras + muestras*lineas*j];//flipa pepinillos !
			}	
		}
		////////----------------------///////
		int newendmember_index[2];
		maxVolumen = 0.0;

		float jointpoint[19*19+19+20] = {};
		creaMatriz(endmember, idx , jdx, ImageIn, n, muestras, lineas, jointpoint);

if(idx == 0 && jdx == 0 && n == 1 && primeraVuelta){//lo hace bien, pero salta muchos core dumps ¿??¿
		//for(a = 0; a < (n+1)*(n+1); a++) mierda[a] = jointpoint[a];
}
		volumen[idx*muestras + jdx] = calculaVolumen(jointpoint, n+1);//se calcula sobre una matriz (n+1)*(n+1)
   

		barrier(CLK_GLOBAL_MEM_FENCE);//esperamos a que todos los hilos calculen el volumen
if(idx == 0 && jdx == 0 && n == 1 && primeraVuelta){
		for(a = 0; a < (n+1)*(n+1); a++) mierda[a] = volumen[a];//el hilo maestro no tiene acceso a todos los volumenes ?????
}

		/*if(idx == 0 && jdx == 0){//REDUCE (sin hacer aun)
			for(i = 0; i < muestras; i++){
				for(j = 0; j < lineas; j++){
					if(volumen[i*muestras+j] > maxVolumen){
						maxVolumen = volumen[i*muestras+j];
						newendmember_index[0] = i;
						newendmember_index[1] = j;
					}
				}
			}

			posiciones[n*2] = newendmember_index[0];
			posiciones[n*2+1] = newendmember_index[1];		
			
			n++;
			if(primeraVuelta){
				n = 1;
				posiciones[0] = newendmember_index[0];
				posiciones[1] = newendmember_index[1];
				primeraVuelta--;
			}
		}else if (primeraVuelta){//modifical una variable local el resto de hilos tienen la suya propia??
			n = 1;
			primeraVuelta--;
		}else */n++;
		

	}//fin del while	
	
}
}

/*
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
				
			}	
		}
		////////----------------------///////
		pos *newendmember_index = (pos*) malloc(1*sizeof(pos));
		float maxVolumen = 0.0;

		for(i = 0; i < muestras; i++){
			for(j = 0; j < lineas; j++){
				//le pasamos el endmember y los puntos i j que vamos a usar, devuelve el array hasta n de ese punto
				float *jointpoint = (float*) malloc ((n*n+n+(n+1))*sizeof(float));;
				creaMatriz(endmember, i , j, imagen, n, muestras, lineas, jointpoint);
	
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

*/

























