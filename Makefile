testNeuralNetwork: testNeuralNetwork.cpp NeuralNetwork.cpp ActivationFunction.cpp CostFunction.cpp
	g++ -o testNeuralNetwork testNeuralNetwork.cpp NeuralNetwork.cpp ActivationFunction.cpp CostFunction.cpp -I . -I /usr/include/eigen3/
