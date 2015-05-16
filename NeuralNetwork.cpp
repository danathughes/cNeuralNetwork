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
#include <iostream>

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
  this->activationFunctions = vector<ActivationFunction>(this->numLayers);
  for(int i=1; i<this->numLayers; i++)
  {
    this->activationFunctions[i] = activationFunction[i];
  }

  this->randomizeWeights();
}


void NeuralNetwork::randomizeWeights()
{
  for(int i=1; i<this->numLayers; i++)
  {
    int fanin = this->weights[i].cols();
    this->weights[i].setRandom();
    this->biases[i].setRandom();
    this->weights[i] = (this->weights[i])/fanin;  // These should really be ranged from -1/fanin to 1/fanin, not 0 to 1
    this->biases[i] = (this->biases[i])/ fanin;
  }
}


double NeuralNetwork::cost(Eigen::MatrixXd dataset, Eigen::MatrixXd targets)
{
  double cost = 0.0;

  return cost;
}


vector<Eigen::MatrixXd> NeuralNetwork::gradient(Eigen::MatrixXd dataset, Eigen::MatrixXd targets)
{
  // Initialize a vector to store the gradients of the weight matrices and biases
  vector<Eigen::MatrixXd> gradients = vector<Eigen::MatrixXd>(2*(this->numLayers - 1));

  return gradients;
}


vector<Eigen::MatrixXd> NeuralNetwork::getWeights()
{
  // Initialize a vector to store all the weights in 
  vector<Eigen::MatrixXd> weightsAndBiases = vector<Eigen::MatrixXd>(2*(this->numLayers - 1));

  for(int i = 0; i < (this->numLayers - 1); i++)
  {
    weightsAndBiases[i] = this->weights[i+1];
    weightsAndBiases[i + (this->numLayers-1)] = this->biases[i+1];
  }

  return weightsAndBiases;
}


void NeuralNetwork::updateWeights(vector<Eigen::MatrixXd> weightUpdate)
{
  for(int i=0; i < (this->numLayers - 1); i++)
  {
    this->weights[i+1] += weightUpdate[i];
    this->biases[i+1] += weightUpdate[i + (this->numLayers - 1)];
  }
}


vector<Eigen::VectorXd> NeuralNetwork::activate(Eigen::VectorXd data)
{
  vector<Eigen::VectorXd> activations = vector<Eigen::VectorXd>(this->numLayers);

  // The activation of the first layer is simply the input
  activations[0] = data;

  return activations;
}
