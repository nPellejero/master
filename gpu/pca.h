#include "funcs.h"

void PCAgpu(int nObjects, void* input, void* output, IplImage* AvgImg, float* eigVals);
int* PCAgpu_i(int nObjects, uchar ** objs,int obj_step,int eig_step,int obj_size_width,int obj_size_height,float* avg_data,int avg_step);
void PCAgpu_cov(int nObjects, uchar** input, float *avg, int avg_step,int size_width,int size__height, float *covarMatrix);
int PCAgpu_eig(float *A, float *V, float *E, int n, float eps);
void PCAgpu_d(int nObjects, uchar** input,float** output,int eig_step,int eig_size_width,
				int eig_size_height,float* avg,float* eigVals, float* ev);
void imprimirMatFloat(CvMat* mat);
void imprimirMat(IplImage* AvgImg);					
void imprimirArrImag(IplImage** arrImg,int n);

