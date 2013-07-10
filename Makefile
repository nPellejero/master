.PHONY : clean, all

all : Faceme

clean :
	-rm  $(objects)

objects = main.o reconocer.o entrenar.o distancia.o \
	pca.o 

Faceme: $(objects)
	g++ -o Faceme $(objects)

main.o : main.cpp funcs.h
	g++ -c main.cpp `pkg-config opencv --cflags --libs`
reconocer.o : reconocer.cpp funcs.h 
	g++ -c reconocer.cpp `pkg-config opencv --cflags --libs`

entrenar.o : entrenar.cpp funcs.h
	g++ -c entrenar.cpp `pkg-config opencv --cflags --libs`
distancia.o : distancia.cpp funcs.h 
	g++ -c distancia.cpp `pkg-config opencv --cflags --libs`
pca.o : pca.cpp funcs.h 
	g++ -c pca.cpp `pkg-config opencv --cflags --libs`

