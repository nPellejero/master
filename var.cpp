#include "funcs.h"

struct dirent **archivos;
struct dirent **carpetas;

IplImage ** arrCaras;                    //Array de imágenes de caras
CvMat    *  numPersonaMat; 	         //Array de ID's de personas
int numCarasEntrenamiento;               //Cantidad de imágenes de entrenamiento
int numEigens;                           //Cantidad de AutoVectores y AutoValores (Eigens)
IplImage * pAvgTrainImg;                 //Imágen promedio
IplImage ** arrAutoVectores;             //AutoVecotores
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
