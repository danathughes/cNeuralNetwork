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

  Eigen::VectorXd target = Eigen::VectorXd(3);
  Eigen::VectorXd output = Eigen::VectorXd(3);
  target(0) = 1.0; target(1) = 0.0; target(2) = 1.0;
  output(0) = 1.0; output(1) = 1.0; output(2) = 0.0;
  cout << "Doing the cost / gradient thing" << endl;
  cout << "Cost: " << net.cost_function(output, target) << endl;
  cout << "Grad: " << net.cost_gradient(output, target) << endl;

  cout << "====== Serious XOR stuff =======" << endl;

  Eigen::MatrixXd dataset = Eigen::MatrixXd(2,4);
  dataset << 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0;
  Eigen::MatrixXd targets = Eigen::MatrixXd(1,4);
  targets << 0.0, 1.0, 1.0, 0.0;

  cout << "Dataset" << endl;
  cout << dataset << endl;
  cout << "Targets" << endl;
  cout << targets << endl;
  cout << "Activations" << endl;
  vector <Eigen::VectorXd> act = net.activate(dataset.col(0));
  cout << act[0] << endl;
  cout << act[1] << endl;
  cout << act[2] << endl;
  cout << "Cost = " << net.cost(act[2],targets.col(0)) << endl;

  cout << "====== Serious Backprop stuff =====" << endl;

  vector<Eigen::MatrixXd> gradient = net.gradient(dataset, targets);

  cout << "I'm back!" << endl;

  for(int i=0; i<gradient.size(); i++)
  {
    cout << gradient[i] << endl;
  }
  

  return 0;
}
