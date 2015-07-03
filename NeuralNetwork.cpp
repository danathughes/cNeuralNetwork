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



Eigen::VectorXd NeuralNetwork::activation_function(Eigen::VectorXd vector)
{
  Eigen::VectorXd results(vector.size());

  for(int i=0; i<vector.size(); i++)
    results(i) = 1.0/(1.0+exp(vector(i)));

  return results;
}

 
Eigen::VectorXd NeuralNetwork::gradient_function(Eigen::VectorXd activations)
{
  Eigen::VectorXd gradients(activations.size());

  for(int i=0; i<activations.size(); i++)
    gradients(i) = activations(i)*(1.0 - activations(i));

  return gradients;
}


double NeuralNetwork::cost_function(Eigen::VectorXd output, Eigen::VectorXd target)
{
  // MSE
  Eigen::VectorXd diff = output-target;
  double cost = 0.5*diff.dot(diff);
  return cost;
}


Eigen::VectorXd NeuralNetwork::cost_gradient(Eigen::VectorXd output, Eigen::VectorXd target)
{
  return target - output;
}



double NeuralNetwork::cost(Eigen::MatrixXd dataset, Eigen::MatrixXd targets)
{
  double cost = 0.0;

  for(int i = 0; i<dataset.cols(); i++)
  {
    cost += this->cost_function(dataset.col(i), targets.col(i));
  }

  return cost / dataset.cols();
}


vector<Eigen::MatrixXd> NeuralNetwork::gradient(Eigen::MatrixXd dataset, Eigen::MatrixXd targets)
{
  // Initialize a vector to store the gradients of the weight matrices and biases
  vector<Eigen::MatrixXd> gradients = vector<Eigen::MatrixXd>(2*this->numLayers);

  gradients[0] = Eigen::MatrixXd(1,1);
  gradients[this->numLayers] = Eigen::VectorXd(1);

  // Set up for calculating the gradients of each weight
  for(int i = 1; i < this->numLayers; i++)
  {
    gradients[i] = Eigen::MatrixXd::Zero(this->weights[i].rows(), this->weights[i].cols());
    gradients[i + this->numLayers] = Eigen::VectorXd::Zero(this->biases[i].rows());
  }


  vector<Eigen::VectorXd> activations;
  vector<Eigen::VectorXd> deltas;
  Eigen::ArrayXd costGradient;
  Eigen::ArrayXd activationGradient;
 
  for(int i = 0; i < dataset.cols(); i++)
  {
    // Do a forward pass to get the activation of each layer
    activations = this->activate(dataset.col(i));
    deltas = vector<Eigen::VectorXd>(this->numLayers-1);

    // Perform backpropagation - there will be the same number of deltas as layers
    // The first delta is the gradient of the cost function (dE/dz) times the gradient of the activation (dz/dy)
    costGradient = this->cost_gradient(activations[this->numLayers-1], targets.col(i)).array();
    activationGradient = this->gradient_function(activations[this->numLayers-1]).array();

    deltas[0] = Eigen::VectorXd(1);    
    deltas[this->numLayers - 1] = (costGradient*activationGradient).matrix();

    // Now do the hidden layers
    for(int j = this->numLayers-1; j > 1; j--)
    {
      deltas[j-1] = this->weights[j].transpose()*deltas[j];
      deltas[j-1] = deltas[j-1].cwiseProduct(this->gradient_function(activations[j-1]));
    }

    // Update the gradients using the forward pass and backpropagation pass
    for(int j = 1; j < this->numLayers; j++)
    {
      gradients[j] -= deltas[j]*(activations[j-1].transpose());
      gradients[j + this->numLayers] -= deltas[j];
    }
    cout << "Done with i=" << i << endl;
    deltas.clear();
    activations.clear();
  }

  // All done!  Divide the gradient by the number of items in the dataset
  for(int i = 1; i < this->numLayers; i++)
  {
    gradients[i] /= dataset.cols();
    gradients[i + this->numLayers] /= dataset.cols();
  } 

  cout << "gradients calculated" << endl;

  return gradients;
}


vector<Eigen::MatrixXd> NeuralNetwork::getWeights()
{
  // Initialize a vector to store all the weights in 
  vector<Eigen::MatrixXd> weightsAndBiases = vector<Eigen::MatrixXd>(2*this->numLayers);

  for(int i = 1; i < this->numLayers; i++)
  {
    weightsAndBiases[i] = this->weights[i];
    weightsAndBiases[i + this->numLayers] = this->biases[i];
  }

  return weightsAndBiases;
}


void NeuralNetwork::updateWeights(vector<Eigen::MatrixXd> weightUpdate)
{
  for(int i = 1; i < this->numLayers; i++)
  {
    this->weights[i] += weightUpdate[i];
    this->biases[i] += weightUpdate[i + this->numLayers];
  }
}


vector<Eigen::VectorXd> NeuralNetwork::activate(Eigen::VectorXd data)
{
  vector<Eigen::VectorXd> activations = vector<Eigen::VectorXd>(this->numLayers);

  // The activation of the first layer is simply the input
  activations[0] = data;

  for(int i=1; i<this->numLayers; i++)
  {
    activations[i] = this->activation_function(this->biases[i] + this->weights[i]*activations[i-1]);
  }

  return activations;
}


Eigen::VectorXd NeuralNetwork::predict(Eigen::VectorXd data)
{
  return this->activate(data)[this->numLayers-1];
}


Eigen::VectorXi NeuralNetwork::classify(Eigen::VectorXd data)
{
  Eigen::VectorXd predictions = this->predict(data);

  double max_val = predictions(0);
  int max_idx = 0;

  for(int i=1; i<predictions.rows(); i++)
  {
    if(max_val < predictions(i))
    {
      max_val = predictions(i);
      max_idx = i;
    }
  }

  Eigen::VectorXi labels = Eigen::VectorXi::Zero(predictions.rows());
  labels(max_idx) = 1;

  return labels;
}

