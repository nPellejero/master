#include "pca.h"

int Gavg_step;
int Gobj_step;
int Gobj_size_width;
int Gobj_size_height;
int Geig_step;
int Geig_size_width;
int Geig_size_height;


/*------------------PRINCIPAL COMPONENTS ANALYSIS------------------*/
void PCA()
{
	int i;
	/*Estructuras*/
	CvTermCriteria calcLimit;
	CvSize tamanoImgCaras;

	/*Cantidad de AutoValores a usar; este es el máximo:*/
	numEigens = numCarasEntrenamiento - 1;

	/*Ajustamos las dimensiones de cada imágen de caras (supuestamente son todas iguales),
	por eso, tomamos la primera del arrCaras (en este momento, en arrCaras, están las caras
	de entrenamiento).*/
	tamanoImgCaras.width  = arrCaras[0]->width;
	tamanoImgCaras.height = arrCaras[0]->height;
	
	/*arrAutoVectores apunta a un array de imágenes IplImage.
	Cuando cvCalcEigenObjects() termina, cada imágen de este array tendrá
	un AutoVector (osea, una AutoCara).*/
	arrAutoVectores = (IplImage**)cvAlloc(sizeof(IplImage*) * numEigens);
	
	arrAutoVectores2 = (IplImage**)cvAlloc(sizeof(IplImage*) * numEigens);
	
	/*Crea en cada celda (desde 0 hasta numEigens - 1) una imágen con tamaño de tamanoImgCaras*/	
	for (i = 0; i < numEigens; i++)
		arrAutoVectores[i] = cvCreateImage(tamanoImgCaras, IPL_DEPTH_32F, 1);
		
	for (i = 0; i < numEigens; i++)
		arrAutoVectores2[i] = cvCreateImage(tamanoImgCaras, IPL_DEPTH_32F, 1);  
	/*Esta matriz va a alojar los AutoValores respectivos a cada AutoVector.
	Los AutoValores son de tipo Float.*/
	matAutoValores = cvCreateMat( 1, numEigens, CV_32FC1 );
	matAutoValores2 = cvCreateMat( 1, numEigens, CV_32FC1 );

	/*Allocate the averaged image:
	To do PCA, the dataset must first be "centered". For our face images,
	this means finding the average image - an image in which each pixel 
	contains the average value for that pixel across all face images in the 
	training set. The dataset is centered by subtracting the average face's 
	pixel values from each training image.
	
	Todo lo anterior sucede dentro de cvCalcEigenObjects(). Esta "imágen promedio" va a estar
	alojada aquí:*/
	pAvgTrainImg = cvCreateImage(tamanoImgCaras, IPL_DEPTH_32F, 1);
	pAvgTrainImg2 = cvCreateImage(tamanoImgCaras, IPL_DEPTH_32F, 1);
	/*Set the PCA termination criterion:
	We tell it to compute each EigenValue, then stop. That's all we need.
	Remember to search more about cvTermCriteria.*/
	calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, numEigens, 1);

	/*Compute average image, eigenvalues, and eigenvectors (this means that'll compute a basis).
	Calcula el subespacio para las caras de entrenamiento*/
	PCAgpu(numCarasEntrenamiento, (void*)arrCaras, (void*)arrAutoVectores, pAvgTrainImg, matAutoValores->data.fl);
	//imprimirMat(pAvgTrainImg);
	//imprimirMatFloat(matAutoValores2);
	//imprimirArrImag(arrAutoVectores2,numEigens);	
	
	/*cvCalcEigenObjects(
		numCarasEntrenamiento,              
		(void*)arrCaras,                   //(input) Donde están guardadas las caras a quienes les calculamos los AutoVectores y AutoValores.
		(void*)arrAutoVectores,            //(output) Donde los guardamos los AutoVectores.
		CV_EIGOBJ_NO_CALLBACK,              //input/output flags
		0,
		0,
		&calcLimit,                         //CvTermCriteria const.
		pAvgTrainImg,                       //Guarda la imágen promedio en pAvgTrainImg.
		matAutoValores->data.fl);
	//imprimirMat(pAvgTrainImg);
	//imprimirMatFloat(matAutoValores);			
	//imprimirArrImag(arrAutoVectores,numEigens);	
	*/
	//printf("ACA\n");		
	cvNormalize(matAutoValores, matAutoValores, 1, 0, CV_L1, 0);

	
	/*cvCalcEigenObjects:
	"The function cvCalcEigenObjects calculates orthonormal eigen basis and the averaged 
	object for a group of the input objects." - Esta función calcula una base ortonormal de los
	AutoVectores que generan el subespacio y, también, calcula el objeto promedio de los input
	(lease: imagen promedio de todas las imagenes de caras del input). -
	Depending on ioFlags parameter it may be used 
	either in direct access or callback mode. Depending on the parameter calcLimit, calculations 
	are finished either after first calcLimit.max_iter dominating eigen objects are retrieved or if 
	the ratio of the current eigenvalue to the largest eigenvalue comes down to calcLimit.epsilon threshold. 
	The value calcLimit -> type must be CV_TERMCRIT_NUMB, CV_TERMCRIT_EPS, or CV_TERMCRIT_NUMB | CV_TERMCRIT_EPS . 
	The function returns the real values calcLimit->max_iter and calcLimit->epsilon .
	
	The function also calculates the averaged object (face image), which must be created previously. 
	Calculated eigen objects are arranged according to the corresponding eigenvalues in the descending order.

	The parameter eigVals may be equal to NULL, if eigenvalues are not needed.*/
	
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
	printf("mat_punt:%p\n mat_height: %d\n mat_width:%d\n mat_mat_step: %d\n",avg_data,avg_size.height,avg_size.width,avg_step);
	for( i = 0; i < avg_size.height; i++, avg_data += avg_step )
		for(int j = 0; j < avg_size.width; j++ )
			printf("%f ",avg_data[j]);
}	

void imprimirArrImag(IplImage** arrImg,int n)
{
	for(int i=0; i< n;i++)
		imprimirMat(arrImg[i]);
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
	
	CvTermCriteria calcLimit;
	calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, nEigens, 1);

	
	
	
	PCAgpu_i( nObjects, objs, obj_step,eigs, eig_step, obj_size.width,obj_size.height, avg_data, avg_step, eigVals);
    
    float* covarMatrix = (float *) cvAlloc( sizeof( float ) * nObjects * nObjects );
    PCAgpu_cov( nObjects, objs, avg_data, Gavg_step, Gobj_size_width,Gobj_size_height, covarMatrix );
       
    float* ev = (float *) cvAlloc( sizeof( float ) * nObjects * nObjects );
    int err = PCAgpu_eig( covarMatrix, ev, eigVals, nObjects, 0.0f );
    assert(err != 1 && err != 2);

	PCAgpu_d(nObjects, objs,eigs,Gobj_size_width,Gobj_size_height,Geig_step,Geig_size_width,Geig_size_height,Gobj_step,Gavg_step,avg_data,eigVals,ev);      


    cvFree( &objs );
    cvFree( &eigs );
	cvFree( &covarMatrix );
	cvFree( &ev );
}


void PCAgpu_i(int nObjects, uchar** objs,int obj_step,float** eigs,int eig_step,int obj_size_width, int obj_size_height ,float* avg_data,int avg_step,float* eigVals)
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
		obj_step =  eig_step = avg_step = obj_size_width;
	}	
	avg_size_width = eig_size_width = obj_size_width;
	avg_size_height = eig_size_height = obj_size_height;

	
	/* Calculation of averaged object */
    bf = avg_data;
    for( i = 0; i < avg_size_height; i++, bf += avg_step )
        for( j = 0; j < avg_size_width; j++ )
            bf[j] = 0.f;
            
    for( i = 0; i < nObjects; i++ )
    {
        int k, l;
        uchar *bu = ((uchar **) objs)[i];

        bf = avg_data;
        assert(avg_step == obj_step);
        for( k = 0; k < avg_size_height; k++, bf += avg_step, bu += obj_step )
            for( l = 0; l < avg_size_width; l++ )
                bf[l] += bu[l];
    }
    bf = avg_data;
    for( i = 0; i < avg_size_height; i++, bf += avg_step )
        for( j = 0; j < avg_size_width; j++ )
            bf[j] *= m;            
    
    
	Gobj_step = obj_step;
	Gavg_step = avg_step;
	Gobj_size_width = obj_size_width;
	Gobj_size_height = obj_size_height;
	Geig_step = eig_step;
	Geig_size_width = eig_size_width;
	Geig_size_height = eig_size_height;
	
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

void PCAgpu_d(int nObjects, uchar** input,float** output,int size_width,int size_height,int obj_step,int avg_step,
					int eig_step,int eig_size_width,int eig_size_height,float* avg,float* eigVals, float* ev)
{
	int m1= nObjects - 1; //max iter
	//float epsilon=0;
	int k, p, l, i;
	float *bf = 0;
	
	 /* Eigen objects number determination */

    for( i = 0; i < m1; i++ )
		if( fabs( eigVals[i] / eigVals[0] ) < EPSILON )
			break;
	m1 = i;
    
    //epsilon = (float) fabs( eigVals[m1 - 1] / eigVals[0] );

    for( i = 0; i < m1; i++ )
        eigVals[i] = (float) (1.0 / sqrt( (double)eigVals[i] ));
        

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
			
			for( p = 0; p < size_height; p++, bu += obj_step, bf += avg_step, be += eig_step )
			{
				for( l = 0; l < size_width - 3; l += 4 )
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
				for( ; l < size_width; l++ )
					be[l] += v * (bu[l] - bf[l]);
					

			}
		}                   /* i */
	}                       /* k */

	for( i = 0; i < m1; i++ )
		eigVals[i] = 1.f / (eigVals[i] * eigVals[i]);

}

