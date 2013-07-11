.PHONY : clean

all : Faceme

clean :
	rm  *.o 

objects = reconocer.o entrenar.o distancia.o \
	pca.o main.o var.o

Faceme: $(objects)
	g++  -o Faceme $(objects) `pkg-config opencv --cflags --libs`

main.o : main.cpp funcs.h 
	g++ -c main.cpp 
	
entrenar.o : entrenar.cpp main.o var.o
	g++ -c  entrenar.cpp 

pca.o : pca.cpp entrenar.o pca.h 
	g++ -c  pca.cpp 	

distancia.o : distancia.cpp main.o
	g++ -c  distancia.cpp

reconocer.o : reconocer.cpp main.o var.o
	g++ -c reconocer.cpp
	 
var.o : funcs.h
	g++ -c var.cpp

eigenobjects.o : eigenobjects.cpp precomp.hpp
	g++ -c eigenobjects.cpp 

