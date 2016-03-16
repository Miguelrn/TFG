
__global double calculaVolumen(__local double *jointpoint_local, double *endmember_vector, int n, int factorial);

#define size 31
#define matrixSize (size+1)*(size+1)


#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable

__kernel void endmembers_calculation(
					__global float* restrict ImageIn,
					__global int* restrict posiciones,
					__global double* restrict volumen,
					const int n,
					const int muestras,
					const int lineas,
					const int bandas,
					__local double* restrict jointpoint_local,
					__global double* restrict jointpoint_aux){


	int i, j;

	double ratio;

	const int lidx = get_local_id(0);
	const int gidx = get_group_id(0)*get_local_size(0) + get_local_id(0);//get global id

	const int lenght = n+1;
	
	double endmember_vector[size+1];	
 
	//if(gidx<muestras*lineas){

		endmember_vector[0] = 1.0;
		for(i = 0; i < n; i++) endmember_vector[i+1] = ImageIn[gidx*bandas+i];//*/ImageIn[gidx + muestras*lineas*i];//gidx*bands+i
		if(get_local_size(0) < lenght * lenght){//se puede quitar y dejar el bucle for solo...
			for(i = lidx; i < lenght * lenght; i += get_local_size(0)){
				if(i < lenght) jointpoint_local[i] = 1.0;
				else jointpoint_local[i] = ImageIn[posiciones[(i%lenght)*2+1] * bandas * muestras + posiciones[(i%lenght)*2] * bandas + (i/lenght-1)];
			}
		}
		else{
			if(lidx < (lenght)*(lenght)){//se puede mejorar
				if(lidx < lenght) jointpoint_local[lidx] = 1.0;
				else jointpoint_local[lidx] = ImageIn[posiciones[(lidx%lenght)*2+1] * bandas * muestras + posiciones[(lidx%lenght)*2] * bandas + (lidx/lenght-1)];
				//ImageIn[posiciones[(lidx%lenght)*2] + posiciones[(lidx%lenght)*2+1]*muestras + muestras*lineas*(lidx/lenght-1)];
				//imagenPAD[s[(i%lenght)].columnas * bands * samples + s[(i%lenght)].filas * bands + (i/lenght-1)]);
			}
		}
		

		int factorial = 1;
		#pragma unroll
		for(i = n; i > 0; i--)
			factorial *= i;

		barrier(CLK_LOCAL_MEM_FENCE); 
		volumen[gidx%muestras*lineas + gidx/muestras] = calculaVolumen(jointpoint_local, endmember_vector, lenght, factorial);


	//}
    	
	
}

__global double calculaVolumen(__local double *jointpoint_local,
				double *endmember_vector,
				int n, 
				int factorial){

	int i = 0,j,k;
	double ratio;

	const int gidx = get_global_id(0);
	const int lidx = get_local_id(0);
	


    	for(i = 0; i < n; i++){//columnas

        	for(j = i+1; j < n; j++){//filas

            		ratio = jointpoint_local[j*n+i]/jointpoint_local[i*n+i];
			barrier(CLK_LOCAL_MEM_FENCE);
			if(lidx < n-1){
				if(get_local_size(0) < n-1){//se puede quitar y dejar el for solo...
					for(k = lidx; k < n-1; k+=get_local_size(0)) jointpoint_local[j*n+k] -= ratio * jointpoint_local[i*n+k];//matrix j k
				}
				else{
					jointpoint_local[j*n+lidx] -= ratio * jointpoint_local[i*n+lidx];//matrix j k
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			endmember_vector[j] -= ratio * endmember_vector[i];
        	}
	}

	barrier(CLK_LOCAL_MEM_FENCE);


	double determinante = 1;
	for(i = 0; i < n-1; i++){
		determinante *= jointpoint_local[i*n+i];	
	}
	determinante *= endmember_vector[n-1];
	determinante = determinante / factorial;
	

	//calculamos el valor absoluto y lo dividimos por el factorial
	if(determinante < 0.0)
		return (-determinante);
	else 
		return determinante;
}


//--------------------------------------------------Segundo Kernel ---------------------------------------------------------------------------//
__kernel void reduce(
			__global int* restrict posiciones,
			__global double* restrict volumen,
			const int n,
			const int primeraVuelta,
			__local double* restrict maxVolumen,
			__local int* restrict posGlobal,
			const int width,
			const int height){

	//const int width = get_global_size(0);
	const int local_width = get_local_size(0);
  	const int gidx = get_global_id(0);
  	const int lidx = get_local_id(0);
	//const int height = widtheight/width;

	int global_index = gidx, i, j;
	double maximo = 0;
	int posicion = 0;
	if(gidx < width*height){
		for(i = lidx; i < width*height; i+=local_width){
			double element = volumen[i];
			if(element > maximo){
				maximo = element;
				posicion = i;
			}
		}

	    	maxVolumen[lidx] = maximo;
		posGlobal[lidx] = posicion;


	   	barrier(CLK_LOCAL_MEM_FENCE);
		for(i = 0; i < local_width; i++){
			if(lidx == i && ((maxVolumen[0] < maxVolumen[i]) || (maxVolumen[0] == maxVolumen[i] && posGlobal[i] < posGlobal[0]) )){
				maxVolumen[0] = maxVolumen[i];
		        	posGlobal[0] = posGlobal[i];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}

	    	if (lidx == 0) {
		
			if(primeraVuelta){
				posiciones[0] = posGlobal[0] / height;
				posiciones[1] = posGlobal[0] % height;

			}
			else{
				posiciones[2*n] = posGlobal[0] / height;
				posiciones[2*n+1] = posGlobal[0] % height;
			}
	    	}
	}
	
}

//--------------------------------------------------Tercer Kernel ---------------------------------------------------------------------------//
__kernel void extrae_endmember(__global int* restrict posiciones,
				__global float* restrict ImageIn,
				__global float* restrict ImageOut,
				const int n,
				const int primeraVuelta,
				const int samples,
				const int lines,
				const int bands){

	const int gidx = get_global_id(0);

	if(primeraVuelta){
		ImageOut[gidx] = ImageIn[posiciones[0] * bands + posiciones[1]*bands*samples + gidx];
	}	
	else{
		ImageOut[n*bands+gidx] = ImageIn[posiciones[2*n] * bands + posiciones[2*n+1]*bands*samples + gidx];
	}

}


