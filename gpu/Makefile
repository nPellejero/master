.PHONY : clean

CXXFLAGS=-g -ggdb -Wall -Wno-unused-function
LDFLAGS=-L/opt/cuda/lib64  -lm  `pkg-config opencv --cflags --libs`

all : Faceme

clean :
	rm  *.o 

objects = reconocer.o entrenar.o distancia.o \
	pca.o main.o var.o pcaGPU.o

Faceme: $(objects)
	g++  -o Faceme $(objects) `pkg-config opencv --cflags --libs`

main.o : main.cpp funcs.h 
	g++ -c main.cpp 
	
entrenar.o : entrenar.cpp pca.o main.o var.o
	g++ -c  entrenar.cpp 

pca.o : pca.cpp pcaGPU.o pca.h pca2.o
	g++ -c  pca.cpp 	


distancia.o : distancia.cpp main.o
	g++ -c  distancia.cpp

reconocer.o : reconocer.cpp main.o var.o
	g++ -c reconocer.cpp
	 
var.o : funcs.h
	g++ -c var.cpp


