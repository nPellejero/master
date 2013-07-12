#include "helpers_cuda.h"
#include "kernel_avg.h"
#include <stdlib.h>


int* PCAgpu_i(int nObjects, unsigned char** objs,int obj_step,int eig_step,int obj_size_width, int obj_size_height ,float* avg_data,int avg_step)
{
	int i,j;

	float m=0;
	float *bf = 0;
    int avg_size_width;
    int avg_size_height;	
    int eig_size_width;
    int eig_size_height;	
	
	avg_step /= 4.0f;
	m = 1.0f/(float)nObjects; //para ahorrar calculos
    eig_step /= 4;
	if( obj_step == obj_size_width && eig_step == obj_size_width && avg_step == obj_size_width )
    {
		obj_size_width *= obj_size_height;
		obj_size_height = 1;
		obj_step = eig_step = avg_step = obj_size_width;
	}	
	avg_size_width = eig_size_width = obj_size_width;
	avg_size_height = eig_size_height = obj_size_height;

	
	/* Calculation of averaged object */
    bf = avg_data;
    for( i = 0; i < avg_size_height; i++, bf += avg_step )
        for( j = 0; j < avg_size_width; j++ )
            bf[j] = 0.f;
    
    //CUDA
    float * avg_gpu;
    size_t avg_size_l = avg_size_height*avg_size_width*sizeof(float); //tamaño de la matriz
    
    CHECK_CUDA_CALL(cudaMalloc(&avg_gpu,avg_size_l ));
    CHECK_CUDA_CALL(cudaMemset(avg_gpu,0,avg_size_l));
    CHECK_CUDA_CALL(cudaMemcpy(avg_data, avg_gpu, avg_size_l, cudaMemcpyDefault));
            
    for( i = 0; i < nObjects; i++ )
    {
        int k, l;
        unsigned char *bu = objs[i];

        bf = avg_data;
        //assert(avg_step == obj_step);
        for( k = 0; k < avg_size_height; k++, bf += avg_step, bu += avg_step )
            for( l = 0; l < avg_size_width; l++ )
                bf[l] += bu[l];
    }
    
    CHECK_CUDA_CALL(cudaMemcpy(avg_data, avg_gpu, avg_size_l, cudaMemcpyDefault));
    mult_gpu(m,avg_gpu,avg_size_height,avg_size_width);
    
    
    bf = avg_data;
    for( i = 0; i < avg_size_height; i++, bf += avg_step )
        for( j = 0; j < avg_size_width; j++ )
            bf[j] *= m;            
    
    //assert(obj_step == avg_step);
    //assert(eig_step == avg_step);


    //ACA objStep = objStep1 = eigStep = eigStep1 = avgStep
    //assert(eig_size_width == obj_size_width);
	//assert(eig_size_height == obj_size_height);

	int* arr = (int*)malloc(3*sizeof(int));
	arr[0] = 0;
	arr[1] = 0;
	arr[2] = 0;
	
		
	arr[0] = avg_step;
	arr[1] = obj_size_width;
	arr[2] = obj_size_height;
	
	return arr;
}



