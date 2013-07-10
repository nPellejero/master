#include "funcs.h"

void PCAgpu(int nObjects, void* input, IplImage** arrOutput, IplImage* AvgImg, float* eigVals) {
	//Variables
	int i,j;
	float m=0;

	IplImage **arrInput = (IplImage **) (((CvInput *) & input)->data);

	float *bf = 0;
	float *avg_data;
    int avg_step = 0;
    CvSize avg_size;

    cvGetImageRawData( AvgImg, (uchar **) & avg_data, &avg_step, &avg_size );

	int obj_step = 0;
	CvSize obj_size = avg_size;
	uchar **objs = (uchar **) cvAlloc( sizeof( uchar * ) * nObjects );

	for( i = 0; i < nObjects; i++ )
    {
        IplImage *img = arrInput[i];
        uchar *obj_data;

        cvGetImageRawData( img, &obj_data, &obj_step, &obj_size );
       
        objs[i] = obj_data;
    }

	avg_step /= 4.0f;
	m = 1.0f/(float)nObjects; //para ahorrar calculos
	if( obj_step == obj_size.width && avg_step == obj_size.width )
    {
		obj_size.width *= obj_size.height;
		obj_size.height = 1;
		obj_step = avg_step = obj_size.width;
	}	
	avg_size = obj_size;
	/* Calculation of averaged object */
    bf = avg_data;
    for( i = 0; i < avg_size.height; i++, bf += avg_step )
        for( j = 0; j < avg_size.width; j++ )
            bf[j] = 0.f;
            
    for( i = 0; i < nObjects; i++ )
    {
        int k, l;
        uchar *bu = ((uchar **) objs)[i];

        bf = avg_data;
        assert(avg_step == obj_step);
        for( k = 0; k < avg_size.height; k++, bf += avg_step, bu += obj_step )
            for( l = 0; l < avg_size.width; l++ )
                bf[l] += bu[l];
    }
    bf = avg_data;
    for( i = 0; i < avg_size.height; i++, bf += avg_step )
        for( j = 0; j < avg_size.width; j++ )
            bf[j] *= m;
}

void PCAgpu_i(int nObjects, void* objs,int obj_step, void* eigs,int eig_step,int obj_size_width,int obj_size_height,float* avg_data,int avg_step,float* eigVals );

void PCAgpu(int nObjects, void* input, void* output, IplImage* AvgImg, float* eigVals) {
	//Variables
	int i;

	int nEigens = nObjects - 1;
	
	IplImage **arrInput = (IplImage **) (((CvInput *) & input)->data);
	IplImage **arrOutput = (IplImage **) (((CvInput *) & output)->data);

	
	float *avg_data;
    int avg_step = 0;
    CvSize avg_size;

    cvGetImageRawData( AvgImg, (uchar **) & avg_data, &avg_step, &avg_size );

	int obj_step = 0;
	int eig_step = 0;
	CvSize obj_size = avg_size, eig_size = avg_size;
	uchar **objs = (uchar **) cvAlloc( sizeof( uchar * ) * nObjects );
	float **eigs = (float **) cvAlloc( sizeof( float * ) * nEigens );
	
	for( i = 0; i < nObjects; i++ )
    {
        IplImage *img = arrInput[i];
        uchar *obj_data;

        cvGetImageRawData( img, &obj_data, &obj_step, &obj_size );
       
        objs[i] = obj_data;
    }
	for( i = 0; i < nEigens; i++ )
    {
		IplImage *eig = arrOutput[i];
        float *eig_data;
				
		cvGetImageRawData( eig, (uchar **) & eig_data, &eig_step, &eig_size );
				
        eigs[i] = eig_data;
    }
    
	PCAgpu_i( nObjects, (void*) objs, obj_step, (void*) eigs, eig_step, obj_size.width,obj_size.height ,avg_data, avg_step, eigVals );
    cvFree( &objs );
    cvFree( &eigs );

}


void PCAgpu_i(int nObjects, void* objs,int obj_step, void* eigs,int eig_step,int obj_size_width,int obj_size_height,float* avg_data,int avg_step,float* eigVals )
{
	float *bf = 0;
	float m=0;
	int i,j;	
	int avg_size_width;
	int avg_size_height;
		
	avg_step = avg_step / 4.0f;
	m = 1.0f/(float)nObjects; //para ahorrar calculos

	if( obj_step == obj_size_width && avg_step == obj_size_width )
    {
		obj_size_width *= obj_size_height;
		obj_size_height = 1;
		obj_step = avg_step = obj_size_width;
	}	
	avg_size_width = obj_size_width;
	avg_size_height = obj_size_height;
	
    bf = avg_data;

    for( i = 0; i < avg_size_height; i++, bf += avg_step )
        for( j = 0; j < avg_size_width; j++ )
            bf[j] = 0.f;

    for( i = 0; i < nObjects; i++ )
    {
        int k, l;
        uchar *bu = ((uchar **) objs)[i];

        bf = avg_data;
        for( k = 0; k < avg_size_height; k++, bf += avg_step, bu += avg_step )
            for( l = 0; l < avg_size_width; l++ )
                bf[l] += bu[l];
    }
 
    bf = avg_data;
    for( i = 0; i < avg_size_height; i++, bf += avg_step )
        for( j = 0; j < avg_size_width; j++ )
            bf[j] *= m;

	/*                
	bf = avg_data;
    for( i = 0; i < avg_size.height; i++, bf += avg_step )
        for( j = 0; j < avg_size.width; j++ )
            printf("%f ",bf[j]);
	*/
}

