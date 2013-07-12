#include "pca.h"

/*------------------PRINCIPAL COMPONENTS ANALYSIS------------------*/
void PCA()
{
	int i;
	/*Estructuras*/
	CvSize tamanoImgCaras;

	/*Cantidad de AutoValores a usar; este es el máximo:*/
	numEigens = numCarasEntrenamiento - 1;

	/*Ajustamos las dimensiones de cada imágen de caras */
	tamanoImgCaras.width  = arrCaras[0]->width;
	tamanoImgCaras.height = arrCaras[0]->height;
	
	arrAutoVectores = (IplImage**)cvAlloc(sizeof(IplImage*) * numEigens);
	
	/*Crea en cada celda (desde 0 hasta numEigens - 1) una imágen con tamaño de tamanoImgCaras*/	
	for (i = 0; i < numEigens; i++)
		arrAutoVectores[i] = cvCreateImage(tamanoImgCaras, IPL_DEPTH_32F, 1);

	/*Esta matriz va a alojar los AutoValores respectivos a cada AutoVector.
	Los AutoValores son de tipo Float.*/
	matAutoValores = cvCreateMat( 1, numEigens, CV_32FC1 );

	pAvgTrainImg = cvCreateImage(tamanoImgCaras, IPL_DEPTH_32F, 1);


	/*Compute average image, eigenvalues, and eigenvectors (this means that'll compute a basis).
	Calcula el subespacio para las caras de entrenamiento*/
	PCAgpu(numCarasEntrenamiento, (void*)arrCaras, (void*)arrAutoVectores, pAvgTrainImg, matAutoValores->data.fl);
		
	cvNormalize(matAutoValores, matAutoValores, 1, 0, CV_L1, 0);
	
	return;
}
void imprimirMatFloat(CvMat* mat)
{
	int alto = mat->rows;
	int ancho = mat-> cols;
	
	printf("tipo: %d\n",mat->type);
	printf("step: %d\n",mat->step);
	printf("row: %d\n",mat->rows);
	printf("cols: %d\n",mat->cols);
	printf("height: %d\n",mat->height);
	printf("width: %d\n",mat->width);
	
	float* dat = mat->data.fl;
	int i,j;
	for (i=0;i<ancho;i++)
		for (j=0;j<alto;j++)
			printf("%f ",dat[i * alto + j]);
			

	printf("\n");
}
	
void imprimirMat(IplImage* AvgImg)
{
	float *avg_data;
	int avg_step = 0,i;
	CvSize avg_size;
   
	cvGetImageRawData( AvgImg, (uchar **) & avg_data, &avg_step, &avg_size );
	avg_step /= 4;

	for( i = 0; i < avg_size.height; i++, avg_data += avg_step )
		for(int j = 0; j < avg_size.width; j++ )
			printf("%f ",avg_data[j]);
}	

void imprimirArrImag(IplImage** arrImg,int n)
{
	for(int i=0; i< n;i++) {
		imprimirMat(arrImg[i]);
		printf("\n");	
	}
}

void PCAgpu(int nObjects, void* input, void* output, IplImage* AvgImg, float* eigVals) {
	//Variables
	int i;
	int nEigens = nObjects - 1;
	 

    IplImage **arrOutput = (IplImage **) (((CvInput *) & output)->data);
	IplImage **arrInput = (IplImage **) (((CvInput *) & input)->data);

	float *avg_data;
    int avg_step = 0;
    CvSize avg_size;

    cvGetImageRawData( AvgImg, (uchar **) & avg_data, &avg_step, &avg_size );

	int obj_step = 0, eig_step = 0;
	
	CvSize obj_size = avg_size, eig_size = avg_size;

    float **eigs = (float **) cvAlloc( sizeof( float * ) * nEigens );	
	uchar **objs = (uchar **) cvAlloc( sizeof( uchar * ) * nObjects );

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
	

	int* arr;
	arr = PCAgpu_i( nObjects, objs, obj_step, eig_step, obj_size.width,obj_size.height, avg_data, avg_step); 
    
    avg_step = arr[0];
	eig_step = avg_step;
	eig_size.width = arr[1];
	eig_size.height = arr[2];
	
    float* covarMatrix = (float *) cvAlloc( sizeof( float ) * nObjects * nObjects );
    PCAgpu_cov( nObjects, objs, avg_data, avg_step, arr[1],arr[2], covarMatrix );
	
    float* ev = (float *) cvAlloc( sizeof( float ) * nObjects * nObjects );
    int err = PCAgpu_eig( covarMatrix, ev, eigVals, nObjects, 0.0f );
    
    assert(err != 1 && err != 2);
	cvFree( &covarMatrix );
	
	PCAgpu_d(nObjects, objs,eigs,eig_step,eig_size.width,eig_size.height,avg_data,eigVals,ev);      

	cvFree( &ev );
    cvFree( &objs );
    cvFree( &eigs );
}



void PCAgpu_cov(int nObjects, uchar** objects, float *avg, int avg_step,int size_width,int size_height, float *covarMatrix)
{
	int obj_step = avg_step;
	int i, j;
	
	
	for( i = 0; i < nObjects; i++ )
    {
		uchar *bu = objects[i];

		for( j = i; j < nObjects; j++ )
		{
			int k, l;
			float w = 0.f;
			float *a = avg;
			uchar *bu1 = bu;
			uchar *bu2 = objects[j];

			for( k = 0; k < size_height; k++, bu1 += obj_step, bu2 += obj_step, a += avg_step )
			{
				for( l = 0; l < size_width - 3; l += 4 )
				{
					float f = a[l];
                    uchar u1 = bu1[l];
                    uchar u2 = bu2[l];

                    w += (u1 - f) * (u2 - f);
                    f = a[l + 1];
                    u1 = bu1[l + 1];
                    u2 = bu2[l + 1];
                    w += (u1 - f) * (u2 - f);
                    f = a[l + 2];
                    u1 = bu1[l + 2];
                    u2 = bu2[l + 2];
                    w += (u1 - f) * (u2 - f);
                    f = a[l + 3];
                    u1 = bu1[l + 3];
                    u2 = bu2[l + 3];
                    w += (u1 - f) * (u2 - f);
				}
                
                for( ; l < size_width; l++ )
                {
					float f = a[l];
                    uchar u1 = bu1[l];
                    uchar u2 = bu2[l];

                    w += (u1 - f) * (u2 - f);
                }
			}

            covarMatrix[i * nObjects + j] = covarMatrix[j * nObjects + i] = w;
        }
    }

}   
	
int PCAgpu_eig(float *A, float *V, float *E, int n, float eps) //metodo rotacion de Jacobi
{
    int i, j, k, ind;
    float *AA = A, *VV = V;
    double Amax, anorm = 0, ax;

    if( A == NULL || V == NULL || E == NULL )
        return 1;
    if( n <= 0 )
        return 2;
    if( eps < 1.0e-7f )
        eps = 1.0e-7f;

    /*-------- Prepare --------*/
    for( i = 0; i < n; i++, VV += n, AA += n )
    {
        for( j = 0; j < i; j++ )
        {
            double Am = AA[j];

            anorm += Am * Am;
        }
        for( j = 0; j < n; j++ )
            VV[j] = 0.f;
        VV[i] = 1.f;
    }

    anorm = sqrt( anorm + anorm );
    ax = anorm * eps / n;
    Amax = anorm;

    while( Amax > ax )
    {
        Amax /= n;
        do                      /* while (ind) */
        {
            int p, q;
            float *V1 = V, *A1 = A;

            ind = 0;
            for( p = 0; p < n - 1; p++, A1 += n, V1 += n )
            {
                float *A2 = A + n * (p + 1), *V2 = V + n * (p + 1);

                for( q = p + 1; q < n; q++, A2 += n, V2 += n )
                {
                    double x, y, c, s, c2, s2, a;
                    float *A3, Apq = A1[q], App, Aqq, Aip, Aiq, Vpi, Vqi;

                    if( fabs( Apq ) < Amax )
                        continue;

                    ind = 1;

                    /*---- Calculation of rotation angle's sine & cosine ----*/
                    App = A1[p];
                    Aqq = A2[q];
                    y = 5.0e-1 * (App - Aqq);
                    x = -Apq / sqrt( (double)Apq * Apq + (double)y * y );
                    if( y < 0.0 )
                        x = -x;
                    s = x / sqrt( 2.0 * (1.0 + sqrt( 1.0 - (double)x * x )));
                    s2 = s * s;
                    c = sqrt( 1.0 - s2 );
                    c2 = c * c;
                    a = 2.0 * Apq * c * s;

                    /*---- Apq annulation ----*/
                    A3 = A;
                    for( i = 0; i < p; i++, A3 += n )
                    {
                        Aip = A3[p];
                        Aiq = A3[q];
                        Vpi = V1[i];
                        Vqi = V2[i];
                        A3[p] = (float) (Aip * c - Aiq * s);
                        A3[q] = (float) (Aiq * c + Aip * s);
                        V1[i] = (float) (Vpi * c - Vqi * s);
                        V2[i] = (float) (Vqi * c + Vpi * s);
                    }
                    for( ; i < q; i++, A3 += n )
                    {
                        Aip = A1[i];
                        Aiq = A3[q];
                        Vpi = V1[i];
                        Vqi = V2[i];
                        A1[i] = (float) (Aip * c - Aiq * s);
                        A3[q] = (float) (Aiq * c + Aip * s);
                        V1[i] = (float) (Vpi * c - Vqi * s);
                        V2[i] = (float) (Vqi * c + Vpi * s);
                    }
                    for( ; i < n; i++ )
                    {
                        Aip = A1[i];
                        Aiq = A2[i];
                        Vpi = V1[i];
                        Vqi = V2[i];
                        A1[i] = (float) (Aip * c - Aiq * s);
                        A2[i] = (float) (Aiq * c + Aip * s);
                        V1[i] = (float) (Vpi * c - Vqi * s);
                        V2[i] = (float) (Vqi * c + Vpi * s);
                    }
                    A1[p] = (float) (App * c2 + Aqq * s2 - a);
                    A2[q] = (float) (App * s2 + Aqq * c2 + a);
                    A1[q] = A2[p] = 0.0f;
                }               /*q */
            }                   /*p */
        }
        while( ind );
        Amax /= n;
    }                           /* while ( Amax > ax ) */

    for( i = 0, k = 0; i < n; i++, k += n + 1 )
        E[i] = A[k];
    /*printf(" M = %d\n", M); */

    /* -------- ordering -------- */
    for( i = 0; i < n; i++ )
    {
        int m = i;
        float Em = (float) fabs( E[i] );

        for( j = i + 1; j < n; j++ )
        {
            float Ej = (float) fabs( E[j] );

            m = (Em < Ej) ? j : m;
            Em = (Em < Ej) ? Ej : Em;
        }
        if( m != i )
        {
            int l;
            float b = E[i];

            E[i] = E[m];
            E[m] = b;
            for( j = 0, k = i * n, l = m * n; j < n; j++, k++, l++ )
            {
                b = V[k];
                V[k] = V[l];
                V[l] = b;
            }
        }
    }

    return 0;
}

void PCAgpu_d(int nObjects, uchar** input,float** output,
					int eig_step,int eig_size_width,int eig_size_height,float* avg,float* eigVals, float* ev)
{
	int m1= 0; //max iter
	//float epsilon=0;
	int k, p, l, i;
	float *bf = 0;
	 /* Eigen objects number determination */

   if(CV_TERMCRIT_ITER != CV_TERMCRIT_NUMBER )
    {
		printf("AAAAAAAAAAAAAAA\n");
        for( i = 0; i < m1; i++ )
		if( fabs( eigVals[i] / eigVals[0] ) < EPSILON )
			break;
		m1 = i;
    }
    else
        m1 = nObjects - 1;
  
    
    //epsilon = (float) fabs( eigVals[m1 - 1] / eigVals[0] );

    for( i = 0; i < m1; i++ )
        eigVals[i] = (float) (1.0f / sqrt( (double)eigVals[i] ));
        

    for( i = 0; i < m1; i++ )       /* e.o. annulation */
    {
		float *be = output[i];

        for( p = 0; p < eig_size_height; p++, be += eig_step )
			for( l = 0; l < eig_size_width; l++ )
				be[l] = 0.0f;
    }

	for( k = 0; k < nObjects; k++ )
	{
		uchar *bv = input[k];

		for( i = 0; i < m1; i++ )
		{
			float v = eigVals[i] * ev[i * nObjects + k];
			float *be = output[i];
			uchar *bu = bv;

			bf = avg;
			
			for( p = 0; p < eig_size_height; p++, bu += eig_step, bf += eig_step, be += eig_step )
			{
				for( l = 0; l < eig_size_width - 3; l += 4 )
				{
					float f = bf[l];
					uchar u = bu[l];

						
					be[l] += v * (u - f);
					f = bf[l + 1];
					u = bu[l + 1];
					be[l + 1] += v * (u - f);
					f = bf[l + 2];
					u = bu[l + 2];
					be[l + 2] += v * (u - f);
					f = bf[l + 3];
					u = bu[l + 3];
					be[l + 3] += v * (u - f);
				}
				for( ; l < eig_size_width; l++ )
					be[l] += v * (bu[l] - bf[l]);
					

			}
		}                   /* i */
	}                       /* k */

	for( i = 0; i < m1; i++ )
		eigVals[i] = 1.0f / (eigVals[i] * eigVals[i]);

}

