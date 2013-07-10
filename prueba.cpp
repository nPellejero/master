#include "funcs.h"

int main() {
	int height,width,step,align,channels;
	uchar *data;
	IplImage* img = 0;
	int j,i,k;
	char name[6] = {"b.png"};
	
	img = cvLoadImage(name, CV_LOAD_IMAGE_UNCHANGED);
	
	// get the image data
	height    = img->height;
	width     = img->width;
	channels  = img->nChannels;
	step      = (img->widthStep / (img->depth/8))/channels ;
	data      = (uchar *)img->imageData;
	align	  = img->align;
	printf("Processing a %dx%d image whit step %d, align %d, depth %d , channels %d, size %d\n",height,width,step,align,img->depth,channels, img->imageSize); 
	
	uchar *data2;
	
	char c;
	scanf("%c",&c);
	
	for(i=0;i<height;i++) 
		for(j=0;j<width;j++) 
			for(k=0;k<channels;k++) {
				data2[i*step+j*channels+k] = data[i*step+j*channels+k];
				printf("data[%d,%d,%d]=%uc ",i,j,k,data2[i*step+j*channels+k]);
			}
	//for(i=900000;i <= img->imageSize+5;i++)
	i=900000;
	while(1) {	
		printf("Data[%d]=%uc ",i,data[i]);
		i++;
	}
	//[(height-1)*step+(width-1)*channels+(channels-1)]
	return 0;
}
