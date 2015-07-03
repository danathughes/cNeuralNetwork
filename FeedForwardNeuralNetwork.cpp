/**
* \class FeedForwardNeuralNetwork
*
* \brief An acylic, feed forward neural network.
*
* \author $Author: dh$
*
* \version $Version: 1.0$
*
* \date $Date: 27-June-2015$
*
* Contact:  danathughes@gmail.com
*
* 
*/

#include "FeedForwardNeuralNetwork.h"
#include <vector>
using namespace std;

FeedForwardNeuralNetwork::FeedForwardNeuralNetwork()
{
  
}

FeedForwardNeuralNetwork::~FeedForwardNeuralNetwork()
{

}

void FeedForwardNeuralNetwork::addInputLayer(Layer* layer)
{
  this->inputLayer = layer;
  this->layers.push_back(layer);
}


void FeedForwardNeuralNetwork::addLayer(Layer* layer)
{
  this->layers.push_back(layer);
}


void FeedForwardNeuralNetwork::addOutputLayer(Layer* layer)
{
  this->layers.push_back(layer);
  this->outputLayer = layer;
}


void FeedForwardNeuralNetwork::addConnection(Connection* connection)
{
  this->connections.push_back(connection);
}


void FeedForwardNeuralNetwork::addBias(Bias* bias)
{
  this->biases.push_back(bias);
}


void FeedForwardNeuralNetwork::setInput(Eigen::VectorXd input)
{
  this->inputLayer->setInput(input);
  this->inputLayer->activate();
}


void FeedForwardNeuralNetwork::setTarget(Eigen::VectorXd target)
{
  this->targetLayer->setInput(target);
  this->targetLayer->activate();
}


void FeedForwardNeuralNetwork::setTargetLayer(Layer* layer)
{
  this->targetLayer = layer;
}


void FeedForwardNeuralNetwork::setObjectiveLayer(ObjectiveLayer* layer)
{
  this->objectiveLayer = layer;
}


void FeedForwardNeuralNetwork::forward()
{
  // Activate the input and target layers - these don't need to 
  // calculate net input
  this->inputLayer->activate();

  // Propagate the activation through the remaining layers
  for(int i=1; i<this->layers.size(); i++)
  {
    layers.at(i)->calculateNetInput();
    layers.at(i)->activate();
  }
}

void FeedForwardNeuralNetwork::backward()
{
  // Start backpropping at the objective layer.  
  for(int i=this->layers.size() - 1; i>0; i--)
  {
    this->layers.at(i)->backprop();
  }
}


vector<Connection*> FeedForwardNeuralNetwork::getConnections()
{
  return this->connections;
}


vector<Bias*> FeedForwardNeuralNetwork::getBiases()
{
  return this->biases;
}

vector<Layer*> FeedForwardNeuralNetwork::getLayers()
{
  return this->layers;
}


Layer* FeedForwardNeuralNetwork::getOutputLayer()
{
  return this->outputLayer;
}


ObjectiveLayer* FeedForwardNeuralNetwork::getObjectiveLayer()
{
  return this->objectiveLayer;
}


Layer* FeedForwardNeuralNetwork::getInputLayer()
{
  return this->inputLayer;
}
