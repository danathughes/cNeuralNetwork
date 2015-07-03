EIGEN_DIR=/usr/include/eigen3/

INCLUDE_DIR=./include
SRC_DIR=./src/
OBJ_DIR=./obj/
BIN_DIR=./bin/


CC=g++
CFLAGS=-I $(EIGEN_DIR) -I $(INCLUDE_DIR)
DEPS = Layer.h FullConnection.h Connection.h Bias.h SigmoidLayer.h LinearLayer.h SoftmaxLayer.h SquaredErrorLayer.h FeedForwardNeuralNetwork.h IdentityConnection.h TanhLayer.h ReLULayer.h Teacher.h SupervisedData.h ObjectiveLayer.h MaxIterationStoppingCriteria.h StoppingCriteria.h AndStoppingCriteria.h GradientDescentTeacher.h

OBJ = Layer.o FullConnection.o Bias.o SigmoidLayer.o LinearLayer.o SoftmaxLayer.o SquaredErrorLayer.o FeedForwardNeuralNetwork.o IdentityConnection.o TanhLayer.o ReLULayer.o Teacher.o SupervisedData.o ObjectiveLayer.o StoppingCriteria.o MaxIterationStoppingCriteria.o AndStoppingCriteria.o GradientDescentTeacher.o xor.o


xor: $(OBJ) 
	$(CC) -o $@ $^ $(CFLAGS)

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: clean

clean:
	rm -f *.o *~ xor
