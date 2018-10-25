run: test main
	./test
	./main
main:
	g++ -std=c++11 -o main main.cpp

test: Model.o
	g++ -std=c++11 -o test ModelTest.cpp Model.o 

Model.o: Model.cpp Model.h
	g++ -c Model.cpp

clean:
	rm *.o test main *.txt