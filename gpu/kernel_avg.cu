#include "helpers_cuda.h"
#include "kernel_avg.h"


static const unsigned int BLOCK_WIDTH = 8;
static const unsigned int BLOCK_HEIGHT = 8;

static const unsigned int GRID_WIDTH = 161;
//static const unsigned int GRID_HEIGHT = 1;


static __global__ void mult(float val,float* matrix, unsigned int height, unsigned int width)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    matrix[x] = matrix[x]*val;
	
}
 

void mult_gpu(float val,float* matrix, unsigned int height, unsigned int width)
{
   // configurar la grilla para el kernel
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(GRID_WIDTH);
    
    mult<<<grid, block>>>(val,matrix,height,width);
}
