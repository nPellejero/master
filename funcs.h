#ifndef FUNCS_H
#define FUNCS_H

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

/*Carga de entorno gráfico*/
IplImage* bienvenida   = cvLoadImage("facemeprints/faceme_presentacion.jpg");
IplImage* salida       = cvLoadImage("facemeprints/faceme_salida.jpg");
IplImage* carga        = cvLoadImage("facemeprints/faceme_cargando.jpg");
IplImage* reconociendo = cvLoadImage("facemeprints/faceme_reconociendo.jpg");
IplImage* final        = cvLoadImage("facemeprints/faceme_exito.jpg");

struct dirent **archivos;
struct dirent **carpetas;

int selecc_arch(const struct dirent *entry);
int selecc_carpetas(const struct dirent *entry);
/*
void entrenar();
void reconocer(char archivoNombresImg[512]);
void PCA();
void guardarDatosEntrenamiento();
int  cargarDatosEntrenamiento(CvMat ** pTrainPersonNumMat);
int  encontrarVecinoCercano(float * projectedTestFace);
int  cargarArrayImgCaras();
int  cargarArrayImgCaras_reconocer(char archivoNombresImg[512]);
void relacion();
void escalado(char * name);
void detectFaces(IplImage * img);
void detectar(IplImage * img);


void ver_imagen(IplImage* imagen);
*/

#endif
