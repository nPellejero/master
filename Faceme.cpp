#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <limits.h>
#include <sys/param.h>
#include <sys/types.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.h>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"


#define FALSO 0
#define VERDADERO !FALSO
#define TAM 80

IplImage ** arrCaras;                    //Array de imágenes de caras
CvMat    *  numPersonaMat; 	         //Array de ID's de personas
int numCarasEntrenamiento;               //Cantidad de imágenes de entrenamiento
int numEigens;                           //Cantidad de AutoVectores y AutoValores (Eigens)
IplImage * pAvgTrainImg;                 //Imágen promedio
IplImage * pAvgTrainImg2;
IplImage ** arrAutoVectores;             //AutoVecotores
IplImage ** arrAutoVectores2;  
CvMat * matAutoValores;                  //AutoValores
CvMat * matImgEntrenamientoProyectadas;  //Imágenes de entrenamiento proyectadas en el subespacio PCA
CvHaarClassifierCascade *cascade_f;		 //Cascada de Clasificacion de Rostros
CvMemStorage            *storage;        //Almacenamiento de informacion
int Aux;                                 //variable auxiliar para saber si se crea una imagen
int Entrenando, Reconociendo, Comenzar;
extern int alphasort();					 //funcion que determina el orden alfabetico, usado para ordenar las fotos de entrenamiento mientras las carga
char trayectoria[MAXPATHLEN];			 // lugar donde se almacenara el directorio actual de trabajo
int AnchoImg;							// Auxiliar para la funcion de escalado
int AltoImg;							// Auxiliar para la funcion de escalado
int ancho;								// Auxiliar para la funcion de escalado
int alto;								// Auxiliar para la funcion de escalado
int caraaux,cara2;

/*Carga de entorno gráfico*/
IplImage* bienvenida   = cvLoadImage("facemeprints/faceme_presentacion.jpg");
IplImage* salida       = cvLoadImage("facemeprints/faceme_salida.jpg");
IplImage* carga        = cvLoadImage("facemeprints/faceme_cargando.jpg");
IplImage* reconociendo = cvLoadImage("facemeprints/faceme_reconociendo.jpg");
IplImage* final        = cvLoadImage("facemeprints/faceme_exito.jpg");

void entrenar();
void reconocer(char archivoNombresImg[512]);
void PCA();
void PCAgpu(int nObjects, void* input, void* arrOutput, IplImage* AvgImg, float* eigVals);
void PCAgpu_i(int nObjects, void* objs,int obj_step, void* eigs,int eig_step,int obj_size_width,int obj_size_height,float* avg_data,int avg_step,float* eigVals );
void guardarDatosEntrenamiento();
int  cargarDatosEntrenamiento(CvMat ** pTrainPersonNumMat);
int  encontrarVecinoCercano(float * projectedTestFace);
int  cargarArrayImgCaras();
int  cargarArrayImgCaras_reconocer(char archivoNombresImg[512]);
void relacion();
void escalado(char * name);
void detectFaces(IplImage * img);
void detectar(IplImage * img);
int selecc_arch(const struct dirent *entry);
int selecc_carpetas(const struct dirent *entry);
struct dirent **archivos;
struct dirent **carpetas;
void ver_imagen(IplImage* imagen);

void presentacion()
{	
	cvNamedWindow("Bienvenido");
	cvMoveWindow("Bienvenido", 300, 0);
	
	cvShowImage("Bienvenido", bienvenida);

	
	while (1)
	{
		if (cvWaitKey(0) == 10)
			break;
		if (cvWaitKey(0) == 27)
			exit(0);
	}
	
	cvDestroyWindow("Bienvenido");
	
	return;
}	



/*------------------MAIN------------------*/

int main(int argc, char** argv)
{
	char* aux = argv[1];
	
	//presentacion();
	
	cvNamedWindow("Cargando...");
	cvMoveWindow("Cargando...", 300, 0);
	
	while (Entrenando == 0)
	{ 
		cvShowImage("Cargando...", carga);
		cvWaitKey(350);
		entrenar();
	}
	
	cvDestroyWindow("Cargando...");
	
	cvNamedWindow("Reconociendo imagen...");
	cvMoveWindow("Reconociendo imagen...", 300, 0);
	
	while (Reconociendo == 0)
	{ 
		cvShowImage("Reconociendo imagen...", reconociendo);
		cvWaitKey(290);
		reconocer(aux);
		system("rm trash/temp.pgm");
	}
	
	
	cvNamedWindow("¡Hasta pronto!");
	cvMoveWindow("¡Hasta pronto!", 300, 0);
	cvShowImage("¡Hasta pronto!", salida);
	
	cvWaitKey(2500);
	
	return 0;
}

/*------------------ENTRENAR------------------*/

void entrenar()
{	
	printf("Entrenando...\n");
	
	int i, offset;

	/*Carga la entrada de entrenamiento.*/
	numCarasEntrenamiento = cargarArrayImgCaras();
	
	if (numCarasEntrenamiento < 2)
	{
		printf("Necesitamos dos o mas caras para el entrenamiento");
		return;
	}

	/*Aplicamos PCA para las caras de entrenamiento.*/
	PCA();
	
	/*Encontramos el subespacio. Ahora convertimos las imágenes de entrenamiento
	en puntos en este subespacio. Se utiliza cvEigenDecomposite().*/

	/*Proyectamos las caras de entrenamiento en el subespacio de PCA.
	Aquí se alojan las proyecciones de las imágenes de entrenamiento.*/
	matImgEntrenamientoProyectadas = cvCreateMat( numCarasEntrenamiento, numEigens, CV_32FC1 );
	
	offset = matImgEntrenamientoProyectadas->step / sizeof(float);
	
	for (i = 0; i < numCarasEntrenamiento; i++)
	{
		/*Proyecta cada imágen de entrenamiento (transformada en un punto) en el
		subespacio creado por PCA.*/
		cvEigenDecomposite(
			arrCaras[i],       //(input)
			numEigens,         //(input)
			arrAutoVectores,   //(input)
			0, 0,
			pAvgTrainImg,
			matImgEntrenamientoProyectadas->data.fl + i*offset);        //(output)
	}
	
	/*The function cvEigenDecomposite calculates all decomposition coefficients 
	for the input object using the previously calculated eigen objects basis and 
	the averaged object. Depending on ioFlags parameter it may be used either in 
	direct access or callback mode.*/
	
	/*Guarda los datos de la función entrenar() en un archivo XML.*/
	guardarDatosEntrenamiento();
	
	Entrenando = 1;
	
	return;
}

/*------------------ETAPA DE RECONOCER------------------*/

void reconocer(char archivoNombresImg[512])
{
	printf("Reconociendo...\n");
	
	int numCarasAReconocer = 0; //Cantidad de caras a reconocer
	int caraEntrenamientoCercana, cercana,cercana2;
	CvMat * trainPersonNumMat = 0; //The person numbers during training
	float * imgAReconocerProyectada = 0;

	

	/*Primer etapa:
	Cargar las caras de las cuales queremos reconocer.
	Es lo mismo que hicimos con las de entrenar*/
	
	numCarasAReconocer = cargarArrayImgCaras_reconocer(archivoNombresImg);
	
	
		
	/*Cargamos la informacion guardada del entrenamiento (el XML)*/
	if( !cargarDatosEntrenamiento( &trainPersonNumMat ) ) 
		return;

	/*Segunda etapa:
	Proyectamos las imágenes a reconocer en el subespacio de PCA
	y localiza la imágen de entrenamiento más cercana. (&)*/
	imgAReconocerProyectada = (float *)cvAlloc(numEigens*sizeof(float));


		/*(&)*/
		cvEigenDecomposite(
			arrCaras[0],
			numEigens,
			arrAutoVectores,
			0, 0,
			pAvgTrainImg,
			imgAReconocerProyectada);
		
		caraEntrenamientoCercana = encontrarVecinoCercano(imgAReconocerProyectada);
		
		if(caraEntrenamientoCercana != 1000)
			cercana = trainPersonNumMat->data.i[caraEntrenamientoCercana];
		else
			cercana = -1;
		
		if(caraaux==1)
			cercana2 = trainPersonNumMat->data.i[cara2];
		else
			cercana2 = -1;
			
		Reconociendo = 1;
		
		cvDestroyWindow("Reconociendo imagen...");
		
		sleep(3);
		
		
		char strAux[99] = "La Persona es: ";
		
		
		cvNamedWindow("Exito");
		cvMoveWindow("Exito", 300, 0);
		
		/*Colocar el texto*/
		CvFont font;
		cvInitFont(&font, CV_FONT_HERSHEY_DUPLEX, 1.0f, 1.0f, 1.0f, 1, CV_AA);
		cvPutText(final, strAux, cvPoint(100, 100), &font, cvScalar(0, 0, 0, 0));
		if(cercana != -1)
			cvPutText(final, carpetas[cercana]->d_name, cvPoint(100, 150), &font, cvScalar(0, 0, 0, 0));
		else	
			cvPutText(final, "DESCONOCIDA", cvPoint(100, 150), &font, cvScalar(0, 0, 0, 0));
			
		if(cercana2 != -1)
			cvPutText(final, carpetas[cercana2]->d_name, cvPoint(100, 200), &font, cvScalar(0, 0, 0, 0));
		//else
			//cvPutText(final, "DESCONOCIDO", cvPoint(100, 200), &font, cvScalar(0, 0, 0, 0));
			
			
		cvShowImage("Exito", final);
		
		cvWaitKey(2222);
	
	
	
	cvDestroyWindow("Exito");
	
	return;
}

/*------------------CARGAR DATOS DE ENTRENAMIENTO------------------*/

int cargarDatosEntrenamiento(CvMat ** pTrainPersonNumMat)
{
	/*Funcion para cargar la info de un archivo XML
	En este caso, necesitamos cargar la info de facedata.xml, donde
	tenemos los valores de numEigens, arrAutoVectores, etc...*/
		
	CvFileStorage * infoArchivo;
	int i;

	/*Abrimos "facedata.xml" con la opción de LEER (CV_STORAGE_READ).*/
	infoArchivo = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_READ );
	
	if (!infoArchivo)
	{
		fprintf(stderr, "No se puede abrir facedata.xml\n");
		return 0;
	}

	/*Lee los datos:*/
	numEigens = cvReadIntByName(infoArchivo, 0, "numEigens", 0);  //Lee el valor titulado como "numEigens" (int nEigens).
	numCarasEntrenamiento = cvReadIntByName(infoArchivo, 0, "numCarasEntrenamiento", 0);
	*pTrainPersonNumMat = (CvMat *)cvReadByName(infoArchivo, 0, "trainPersonNumMat", 0);  //ID's
	matAutoValores  = (CvMat *)cvReadByName(infoArchivo, 0, "matAutoValores", 0);
	matImgEntrenamientoProyectadas = (CvMat *)cvReadByName(infoArchivo, 0, "matImgEntrenamientoProyectadas", 0);
	pAvgTrainImg = (IplImage *)cvReadByName(infoArchivo, 0, "avgTrainImg", 0);
	arrAutoVectores = (IplImage **)cvAlloc(numCarasEntrenamiento*sizeof(IplImage *));
	for (i = 0; i < numEigens; i++)  //Lee lols AutoVectores
	{
		char varname[200];
		sprintf(varname, "eigenVect_%d", i);
		arrAutoVectores[i] = (IplImage *)cvReadByName(infoArchivo, 0, varname, 0);
	}

	cvReleaseFileStorage( &infoArchivo );

	return 1;
}

/*------------------GUARDAR DATOS DE ENTRENAMIENTO------------------*/

void guardarDatosEntrenamiento()
{
	CvFileStorage * infoArchivo;
	int i;

	/*Crea un "apuntador" al archivo XML.
	CV_STORAGE_WRITE: crear y/o escribir en el archivo que señalamos "facedata.xml".*/
	infoArchivo = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_WRITE );

	/*Guardamos toda la info allí (cvWrite)*/
	cvWriteInt( infoArchivo, "numEigens", numEigens );  //"nEigens le ponemos como nombre a nEigens.
	cvWriteInt( infoArchivo, "numCarasEntrenamiento", numCarasEntrenamiento );
	cvWrite(infoArchivo, "trainPersonNumMat", numPersonaMat, cvAttrList(0,0));
	cvWrite(infoArchivo, "matAutoValores", matAutoValores, cvAttrList(0,0));
	cvWrite(infoArchivo, "matImgEntrenamientoProyectadas", matImgEntrenamientoProyectadas, cvAttrList(0,0));
	cvWrite(infoArchivo, "avgTrainImg", pAvgTrainImg, cvAttrList(0,0));
	for (i = 0; i < numEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		cvWrite(infoArchivo, varname, arrAutoVectores[i], cvAttrList(0,0));
	}

	cvReleaseFileStorage(&infoArchivo);
	
	return;
}

/*------------------ENCONTRAR EL VECINO (punto) MAS CERCANO------------------*/

int encontrarVecinoCercano(float * imgAReconocerProyectada)
{
	/*Eigenface "recognizes" a face image by looking for the training 
	image that's closest to it in the PCA subspace. Finding the closest 
	training example in a learned subspace is a very common AI technique. 
	It's called Nearest Neighbor matching.*/
	
	/* It computes distance from the Projected Test Image to each projected
	training example. The distance basis here is "Squared Euclidean Distance."*/
 
	/*Trabajamos en un Espacio Euclideano. La distancia entre dos puntos es
	el módulo del vector mismo. El módulo es el usual: la raiz cuadrada de 
	la suma de las componentes del vector al cuadrado.
	Acá solamente tomamos la suma y obviamos la raíz ya que el módulo (distancia
	entre dos puntos) de un vector v es menor que el módulo de un vector u sii
	la suma de los compontentes de v al cuadrado es menor que las sumas de u.*/
	
	CvMat * trainPersonNumMat = 0;
	/*Cargamos la informacion guardada del entrenamiento (el XML)*/
	if( !cargarDatosEntrenamiento( &trainPersonNumMat ) ) 
		perror("cargarDatosEntrenamiento");
	
	double minDist = DBL_MAX;  //Max double
	double minDist2 = DBL_MAX;
	int i, nunCaraEntrenada, caraEntrenamientoCercana = 0,porcentaje;
	double dist = 0;
	float resta = 0;

	for (nunCaraEntrenada = 0; nunCaraEntrenada < numCarasEntrenamiento; nunCaraEntrenada++)
	{
		dist = 0;

		for (i = 0; i < numEigens; i++)
		{
			/*Hace el punto (en el subespacio PCA) proyectado de cada imágen
			a reconocer menos el punto proyectado (en el subespacio PCA) de cada
			imágen de entrenamiento. Luego, a esta resta, le aplica el cuadrado.
			Por lo tanto, es el módulo de la distancia entre el punto proyectado de 
			cada imágen a reconocer y el punto proyectado de cada imágen de entrenamiento.*/
			
			resta = imgAReconocerProyectada[i] - matImgEntrenamientoProyectadas->data.fl[nunCaraEntrenada*numEigens + i];
			dist += resta*resta; // Euclidean
			//dist += resta*resta / matAutoValores->data.fl[i];  // Mahalanobis
			
			//printf(" %f \n",dist);
			
		}
		dist = sqrt(dist);
		//printf(" DISTANCIA = %f \n",dist);
		
		/*Encuentra la mínima distancia:
		Nos dice qué cara de entrenamiento es la que mejor coincide con las caras
		a reconocer*/
		if (dist < minDist)
		{
			if (trainPersonNumMat->data.i[caraEntrenamientoCercana] != trainPersonNumMat->data.i[nunCaraEntrenada])
			{
				minDist2 = minDist;
				cara2 = caraEntrenamientoCercana;
			}
			minDist = dist;
			caraEntrenamientoCercana = nunCaraEntrenada;
			
		}
		
		
		/*double maxDist = DBL_MAX;
		if (dist > maxDist)
			maxDist = dist;
				
		printf(" Maxima = %f \n",maxDist); */
	}
	//printf(" cara: %d, cara2 %d \n", caraEntrenamientoCercana, cara2);
	//printf(" cara: %d, cara2 %d \n",trainPersonNumMat->data.i[caraEntrenamientoCercana], trainPersonNumMat->data.i[cara2]);
	if(minDist > 7500) //IMAGEN DESCONOCIDA
	{
		printf("DESCONOCIDO \n");
		return 1000; 	//caraEntrenamientoCercana = 1000
	}
	
	
	if(minDist < 1000) //LA IMAGEN ESTÁ EN LA BASE DE DATOS
		porcentaje = 100;
	else
	{
		if(minDist <1500)
			porcentaje = 90;
		else
			porcentaje = 100 - (minDist*90)/7500 ;
	}	
	printf(" MINIMA = %f \n",minDist);
	printf("Porcentaje:  %d \n",porcentaje);
	
	if(5000 < minDist && minDist < 7500)
	{
		caraaux = 1;
		printf("Al ser bajo el porcentaje es poca la probabilidad del acierto\n ");
	}
	
	
	/*Devuelve la cara de entrenamiento más cercana.*/
	return caraEntrenamientoCercana;
}

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

	printf("caras = %d\n",numCarasEntrenamiento);
	PCAgpu(numCarasEntrenamiento, (void*)arrCaras, (void*) arrAutoVectores2, pAvgTrainImg2, matAutoValores->data.fl);

	//printf("caras = %d\n",numCarasEntrenamiento);

	cvCalcEigenObjects(
		numCarasEntrenamiento,              
		(void*)arrCaras,                   //(input) Donde están guardadas las caras a quienes les calculamos los AutoVectores y AutoValores.
		(void*)arrAutoVectores,            //(output) Donde los guardamos los AutoVectores.
		CV_EIGOBJ_NO_CALLBACK,              //input/output flags
		0,
		0,
		&calcLimit,                         //CvTermCriteria const.
		pAvgTrainImg,                       //Guarda la imágen promedio en pAvgTrainImg.
		matAutoValores->data.fl);

	//printf("caras = %d\n",numCarasEntrenamiento);
/*
	float *bf = 0;
	float *avg_data;
    int avg_step = 0;
    CvSize avg_size;
    
	cvGetImageRawData( pAvgTrainImg, (uchar **) & avg_data, &avg_step, &avg_size );
	bf = avg_data;
	for( i = 0; i < avg_size.height; i++, bf += avg_step )
        for(int j = 0; j < avg_size.width; j++ )
            printf("%f ",bf[j]);
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
  //  printf("w %d, h %d \n",obj_size.width,obj_size.height);
	PCAgpu_i( nObjects, (void*) objs, obj_step, (void*) eigs, eig_step, obj_size.width,obj_size.height ,avg_data, avg_step, eigVals );
    cvFree( &objs );
    cvFree( &eigs );

}


void PCAgpu_i(int nObjects, void* objs,int obj_step, void* eigs,int eig_step,int obj_size_width,int obj_size_height,float* avg_data,int avg_step,float* eigVals )
{
	float *bf = 0;
	float m=0;
	int i,j;	
	int avg_size_width = 0;
	int avg_size_height = 0;
		
	avg_step /= 4;
	m = 1.0f/(float)nObjects; //para ahorrar calculos
	//printf("w %d, h %d \n",obj_size_width,obj_size_height);
	if( obj_step == obj_size_width && avg_step == obj_size_width )
    {
		obj_size_width *= obj_size_height;
		obj_size_height = 1;
		avg_step = obj_size_width;
	}	
	avg_size_width = obj_size_width;
	avg_size_height = obj_size_height;
	
    bf = avg_data;
	//printf("s %d \n",avg_step);
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
	
                
	bf = avg_data;
    for( i = 0; i < avg_size_height; i++, bf += avg_step )
        for( j = 0; j < avg_size_width; j++ )
            printf("%f ",bf[j]);
	
	
}



/*------------------INICIALIZAR EL ARRAY DE CARAS (arrCaras)------------------*/

int cargarArrayImgCaras()
{
	/*Carga las imágenes y los ID de cada persona en los casos de "entrenar" y "reconocer".*/
	
	char cadena[MAXPATHLEN];
	int contar, contar2, contar3;
	int i, j, aux=0, numCaras = 0;
	//char archivoNombresImg[512];   //Buffer donde guarda caracteres
	
	if ( getcwd(trayectoria,TAM) == NULL )
    { 
		printf("Error obteniendo la trayectoria actual\n");
        exit(0);
    }
    
    contar = scandir(trayectoria, &carpetas, selecc_carpetas, alphasort);
	
	if (contar <= 0)
    { 
        printf("No hay archivos en este direntorio\n");
        exit(0);
    }
    
    for (i=0; i<contar; ++i)
    {
		chdir(carpetas[i]->d_name);
						
		if ( getcwd (cadena,TAM) == NULL )
		{ 
			printf("Error obteniendo la trayectoria actual\n");
			exit(0);
		}
		
		contar2 = scandir(cadena, &archivos, selecc_arch, alphasort);
		
		if (contar2 <= 0)
		{ 
			printf("No hay archivos en este direntorio\n");
			exit(0);
		}
		
		numCaras += contar2;
		chdir("..");
	}
	
	
	/*Inicializaciones*/
	arrCaras        = (IplImage **)cvAlloc( numCaras*sizeof(IplImage *) );
	numPersonaMat = cvCreateMat( 1, numCaras, CV_32SC1 );
	
	for (i=0; i<contar; ++i)
	{
		chdir(carpetas[i]->d_name);
						
		if ( getcwd (cadena,TAM) == NULL )
		{ 
			printf("Error obteniendo la trayectoria actual\n");
			exit(0);
		}
		
		contar3 = scandir(cadena, &archivos, selecc_arch, alphasort);
		
		/*Colocar cada imágen (cara) dentro de cada celda de arrCaras*/
		for (j=0; j < contar3; j++)
		{
			numPersonaMat->data.i[aux + j] = i;	 //data.i (ya que el ID es un int)
			
			escalado(archivos[j]->d_name);// redimensiona la imagen de nombre archivoNombresImg
			
			arrCaras[aux + j] = cvLoadImage(archivos[j]->d_name, CV_LOAD_IMAGE_GRAYSCALE); //cargamos la imagen en escala de grises
			//arrCaras[aux + j] = cvLoadImage(archivos[j]->d_name, CV_LOAD_IMAGE_UNCHANGED); //cargamos la imagen sin cambios
		
			//Por si no pudo cargar la imágen:
			if (!arrCaras[aux + j])
			{
				fprintf(stderr, "No se pudo cargar la imágen %s\n", archivos[j]->d_name);
				return 0;
			}
		
			//Enfoca
			cvEqualizeHist(arrCaras[aux + j],arrCaras[aux + j]);
		
		}
				
		aux += contar3;
					
		chdir("..");
	}
	
	
	return numCaras;
}


int cargarArrayImgCaras_reconocer(char archivoNombresImg[512]) /*archivoNombresImg tiene la ruta de la imágen a reconocer*/
{
	//Carga la imágen a reconocer
	int aux = 0, numCaras = 1;
	char temp[] = "trash/temp.pgm";
	
	/*Inicializaciones*/
	arrCaras        = (IplImage **)cvAlloc( numCaras*sizeof(IplImage *) );

	IplImage* img_aux_original = cvLoadImage(archivoNombresImg);
	
	//Por si no pudo cargar la imágen:
		if (!img_aux_original)
		{
			fprintf(stderr, "No se pudo cargar la imágen %s\n",archivoNombresImg);
			return 0;
		}
	
	cvSaveImage(temp,img_aux_original);
	cvReleaseImage(&img_aux_original);
	
	//IplImage* img_aux = cvLoadImage(temp);
	//detectar(img_aux);
	
	
		
	/*Carga las imágenes de caras:
	arrCaras es un array donde en cada celda hay una imágen (cara).
	La transforma en GrayScale.*/
	
	if (Aux == 1) //Esto pasa si usamos el detector de caras: detectFaces
	{ 
		char auxiliar[] = "trash/imagenCaraRecortada.pgm";
		escalado(auxiliar);// redimensiona la imagen de nombre archivoNombresImg
		arrCaras[aux] = cvLoadImage(auxiliar, CV_LOAD_IMAGE_GRAYSCALE);  //cargamos la imagen en escala de grises
		//arrCaras[aux] = cvLoadImage(auxiliar, CV_LOAD_IMAGE_UNCHANGED); //cargamos la imagen sin cambios
		
		//Por si no pudo cargar la imágen:
		if (!arrCaras[aux])
		{
			fprintf(stderr, "No se pudo cargar la imágen %s\n", auxiliar);
			return 0;
		} 
		
		cvEqualizeHist(arrCaras[aux],arrCaras[aux]);
		//ver_imagen(arrCaras[aux]);
	}	
	else
	{
		escalado(temp);// redimensiona la imagen de nombre archivoNombresImg
		
		arrCaras[aux] = cvLoadImage(temp, CV_LOAD_IMAGE_GRAYSCALE); //cargamos la imagen en escala de grises
		//arrCaras[aux] = cvLoadImage(archivoNombresImg, CV_LOAD_IMAGE_UNCHANGED); //cargamos la imagen sin cambios
		
		
		//Por si no pudo cargar la imágen:
		if (!arrCaras[aux])
		{
			fprintf(stderr, "No se pudo cargar la imágen %s\n", temp);
			return 0;
		}
			
		cvEqualizeHist(arrCaras[aux],arrCaras[aux]);
		//ver_imagen(arrCaras[aux]);	
	}
	
	return numCaras;
}

void ver_imagen(IplImage* imagen)
{
	cvNamedWindow( "test", 1); // representamos la imagen escalada 
                 // (con el 1 indicamos que la ventana se ajuste a los parámetros de la imagen)
 
	cvShowImage( "test", imagen); 
	cvWaitKey(0); // pulsamos cualquier tecla para terminar el programa
	cvDestroyAllWindows(); // destruimos todas las ventanas
}

void relacion()
{
	float rel;
	
	rel = ((float)ancho/(float)alto);
	//printf("relacion = %f \n", rel);
	
	if (rel < 0.81 or 0.83 < rel)
	{
		if(rel < 0.81)
		{
			alto = alto - 10;
			AltoImg = AltoImg + 10;
			relacion();
		}
		if(rel > 0.83)
		{
			ancho = ancho - 10;
			AnchoImg = AnchoImg + 10;
			relacion();
		}
	}

}

void escalado(char * name)
{
	IplImage* imagen; //Inicialización de "imagen" 
	IplImage* img = NULL;
	
	img = cvLoadImage(name, 0); //Cargamos la imagen
	
	/*Píxeles en el eje x de la imagen escalada, es decir, estamos definiendo la escala X*/
	int px = 92;
	/*Píxeles en el eje x de la imagen escalada, es decir, estamos definiendo la escala Y*/
	int py = 112;
	
	ancho = img->width;
	alto = img->height;
	
	if(0.81 > (float)ancho/(float)alto or 0.83 < (float)ancho/(float)alto )
	{ 
		relacion();
	
		cvSetImageROI( img , cvRect( AnchoImg/2 , 1 , ancho, alto) ) ;
	
		imagen = cvCreateImage( cvGetSize(img) ,img->depth, img->nChannels);
	
		cvCopy ( img , imagen , NULL ) ;

		cvResetImageROI(img);
		cvReleaseImage(&img);
 
	}
	else
		imagen = cvLoadImage(name, 0); //Cargamos la imagen
		
	/*Creamos la estructura donde ubicaremos la imagen escalada, 
    siendo px y py los píxeles de la imagen destino, es decir, 
    el propio factor de escala.*/
	IplImage *resized = cvCreateImage(cvSize(px, py), IPL_DEPTH_8U, 1); 
	
	cvResize(imagen, resized,CV_INTER_LINEAR); //Función escalado de imagen
	
	//cvEqualizeHist(resized, resized);
	
	cvSaveImage(name, resized);

	
	
	cvReleaseImage(&imagen);
	cvReleaseImage(&resized);

}

void detectar(IplImage * img)
{
	// Archivos de cascada de caracteristicas para ...
    const char *file1 = "haarcascade_frontalface_alt.xml"; // Deteccion de Rostros
   
    /* Cargar Clasificador de Rostros */
    cascade_f = (CvHaarClassifierCascade*)cvLoad(file1, 0, 0, 0);
    /* Inicializar el Modulo de Memoria, Necesario para el Detector de Rostros */
    storage = cvCreateMemStorage(0);
 
	detectFaces(img);
   
	cvReleaseImage( &img );
      
}
 
void detectFaces(IplImage * img) 
{
  
	CvSeq* faces = cvHaarDetectObjects( img, cascade_f, storage,1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(20, 20) );
 
    for( int i = 0; i <(faces ? faces->total : 0); i++ ) 
    {
        CvRect *r = (CvRect*)cvGetSeqElem( faces, i );
        cvRectangle( img, cvPoint(r->x,r->y),cvPoint(r->x+r->width,r->y+r->height),CV_RGB(255,0,0), 3 );
        cvSetImageROI (img, *r);
        
        /* Crear imagen de destino
		Tener en cuenta que cvGetSize devolverá la anchura y la altura del retorno de la inversión  */
		IplImage *img2 = cvCreateImage( cvGetSize(img) , img->depth, img->nChannels) ;
 
		/* Copia sub-imagen */
		cvCopy ( img , img2 , NULL ) ;
        
        if (img2->width < 91 && img2->height < 111)
        {
			cvResetImageROI(img);
			cvReleaseImage(&img2);
		}
        else
        {
			cvSaveImage( "trash/imagenCaraRecortada.pgm" ,img2,0); //Cambié de carpeta
			printf("creo la imagen \n");
			Aux = 1;
			/* Siempre restablecer la Región de Interés */
			cvResetImageROI(img);
			cvReleaseImage(&img2);
		}
    }
 
    cvClearMemStorage(storage);
}

int selecc_carpetas(const struct dirent *entry)
{ 
	if ((strcmp(entry->d_name, ".") == 0)  || (strcmp(entry->d_name, "..") == 0) || (strcmp(entry->d_name, "facemeprints") == 0) || (strcmp(entry->d_name, "trash") == 0) || (strcmp(entry->d_name, "Info proyecto") == 0) )
        return (FALSO);
        
    if(entry->d_type & DT_DIR)
		return (VERDADERO);
	else
		return (FALSO);
}
int selecc_arch(const struct dirent *entry)
{
	const char *ptr;
                
    if ((strcmp(entry->d_name, ".")== 0) || (strcmp(entry->d_name, "..") == 0))
        return (FALSO);
    
    
    ptr = rindex(entry->d_name, '.'); // Probar que tenga un punto 
    
    if ( (ptr != NULL )  && ( ((strcmp(ptr, ".pgm") == 0)) || ((strcmp(ptr, ".jpg") == 0)) || ((strcmp(ptr, ".JPG") == 0)) ) )//|| (strcmp(ptr, ".h") == 0) || (strcmp(ptr, ".o") == 0) ) )
        return (VERDADERO);
    else
        return(FALSO);

}


