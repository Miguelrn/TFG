#define BLK 512

__kernel void pixel_projections(__global double *imageIn,
				__global double *proymatrix,
				const int linessamples,
				const int Nmax,
				__global double *imageOut){

	int i,j;
	const int gidx = get_global_id(0);
	double value = 0.0, value_fin=0.0;

	if(gidx < linessamples){
		for (i = 0; i < Nmax; i++){
			value = 0.0;
			for (j = 0; j < Nmax; j++){
				value += proymatrix[j + i*Nmax] * imageIn[gidx + j*linessamples];
			}

			value_fin += value*value;
		}
		imageOut[gidx] = value_fin;	
	}	

}

	
__kernel void max_bright(__global double *imageIn,
			const int linessamples,
			const int bands,
			__global double *imageOut){

	const int gidx = get_global_id(0);
	int i = 0;
	__local double value;
	value = 0.0;

	if(gidx < linessamples){
		imageIn[gidx+linessamples*(bands-1)] = 1;//fusion de dos kernels
		for(i = 0; i < bands; i++)
			value += imageIn[gidx + linessamples*i];
		imageOut[gidx] = value*value;
	}

}

__kernel void max_bright_reduce(__global double *imageIn,
				const int linessamples,
				__global int *indice,
				__global double *projection){

	__local double local_projection[BLK];
	__local int local_indice[BLK];

	const int gidx = get_global_id(0);
	const int lidx = get_local_id(0);
	
	int global_index;
	double maximo = 0;
	int posicion = 0;
	if(gidx < linessamples){
	
		for(global_index = gidx; global_index < linessamples; global_index+=get_global_size(0)){
			if(imageIn[global_index] > maximo){
				maximo = imageIn[global_index];
				posicion = global_index;
			}
		}

		local_projection[lidx] = maximo;
		local_indice[lidx] = posicion; 
  		barrier(CLK_LOCAL_MEM_FENCE);

		for(int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
    			if (lidx < offset) {
				if(local_projection[lidx + offset] > local_projection[lidx]){
					local_projection[lidx] = local_projection[lidx + offset];
					local_indice[lidx] = local_indice[lidx + offset];
				}
    			}
    			barrier(CLK_LOCAL_MEM_FENCE);
  		}


		if (lidx == 0) {
    			projection[get_group_id(0)] = local_projection[0];
			indice[get_group_id(0)] = local_indice[0];
  		}

	}
	

}


__kernel void mean_pixel(__global double *imageInOut,
			const int linessamples,
			const int bandas,
			__local double *sdata){


	int i, j;
	const int gidx = get_global_id(0);
	const int lidx = get_local_id(0);
	const int local_size = get_local_size(0);


	for(i = 0; i < bandas; i++){
		sdata[lidx] = 0;
		for(j = lidx; j < linessamples; j += local_size){//reduccion en dos etapas
			sdata[lidx] += imageInOut[i*linessamples + j];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		for(j = local_size / 2; j > 0; j = j / 2) {
			if (lidx < j) {
				sdata[lidx] += sdata[lidx + j];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		if(lidx == 0) sdata[lidx] /= linessamples;
		barrier(CLK_LOCAL_MEM_FENCE);

		for(j = lidx; j < linessamples; j += local_size){
			imageInOut[i*linessamples + j] -=  sdata[0];
		}
		
		
		
	}

}


__kernel void prueba(__global double *A){
	const int gidx = get_global_id(0);
	
	A[gidx] = 2*A[gidx];
}


