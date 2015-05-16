/**
* testNeuralNetwork.cpp
*/

#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "CostFunction.h"

#include <Eigen/Dense>

#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char** argv)
{
  vector<int> layers = vector<int>(3);
  vector<ActivationFunction> activations = vector<ActivationFunction>(3);
  layers[0] = 2; layers[1] = 2; layers[2] = 1;
  activations[1] = ActivationFunction(); activations[2] = ActivationFunction();
  CostFunction costFunction = CostFunction();
  
  cout << layers[0] << " " << layers[1] << " " << layers[2] << endl;

  cout << "Testing Neural Network Functionality" << endl;

  cout << "Creating a new neural network" << endl;
  NeuralNetwork net = NeuralNetwork(layers, activations, costFunction);
 
  cout << "Weights" << endl;
  vector<Eigen::MatrixXd> weights = net.getWeights();

  for(int i = 0; i<4; i++)
    cout << weights[i] << endl << endl;

  cout << "Activating on [0,1]" << endl;

  

  return 0;
}
