/**
* NeuralNetwork.cpp
* Dana Hughes
* version 1.0
* 15-May-2015
* 
* Fully connected neural network model, which predicts an output vector given
* an input vector.
*
* Revisions:
*   1.0		Initial version, ported from the python version.
*/

#include "NeuralNetwork.h"

#include <vector>

//#include <Eigen/Dense>

using namespace std;

/**
*
*/

NeuralNetwork::NeuralNetwork(std::vector<int> layerSizes, vector<ActivationFunction> activationFunction, CostFunction costFunction) 
{
  // Store the input, output and number of layers
  this->numLayers = layerSizes.size();
  this->numInputs = layerSizes[0];
  this->numOutputs = layerSizes[layerSizes.size() - 1];
  this->layerSizes = layerSizes;

  // Initialize the weights and biases
  this->weights = vector<Eigen::MatrixXd>(this->numLayers);
  this->biases = vector<Eigen::VectorXd>(this->numLayers);

  for(int i=1; i<this->numLayers; i++)
  {
    this->weights[i] = Eigen::MatrixXd(this->layerSizes[i], this->layerSizes[i-1]);
    this->biases[i] = Eigen::VectorXd(this->layerSizes[i]);
  }

  // Assign the activation and cost functions

  this->costFunction = costFunction;

}
