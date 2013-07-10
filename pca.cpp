#include "funcs.h"
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
	
	/*Crea en cada celda (desde 0 hasta numEigens - 1) una imágen con tamaño de tamanoImgCaras*/	
	for (i = 0; i < numEigens; i++)
		arrAutoVectores[i] = cvCreateImage(tamanoImgCaras, IPL_DEPTH_32F, 1);

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

	/*Set the PCA termination criterion:
	We tell it to compute each EigenValue, then stop. That's all we need.
	Remember to search more about cvTermCriteria.*/
	calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, numEigens, 1);

	/*Compute average image, eigenvalues, and eigenvectors (this means that'll compute a basis).
	Calcula el subespacio para las caras de entrenamiento*/
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
