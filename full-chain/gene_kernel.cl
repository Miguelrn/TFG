

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



