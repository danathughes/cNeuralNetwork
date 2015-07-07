/**
* \class RecurrentNeuralNetwork
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

#include <iostream>

#include "RecurrentNeuralNetwork.h"
#include <vector>
using namespace std;

RecurrentNeuralNetwork::RecurrentNeuralNetwork()
{
  
}

RecurrentNeuralNetwork::~RecurrentNeuralNetwork()
{

}

void RecurrentNeuralNetwork::addInputLayer(Layer* layer)
{
  this->inputLayer = layer;
  this->layers.push_back(layer);
}


void RecurrentNeuralNetwork::addLayer(Layer* layer)
{
  this->layers.push_back(layer);
}


void RecurrentNeuralNetwork::addOutputLayer(Layer* layer)
{
  this->layers.push_back(layer);
  this->outputLayer = layer;
}


void RecurrentNeuralNetwork::addConnection(Connection* connection)
{
  this->connections.push_back(connection);
}


void RecurrentNeuralNetwork::addBias(Bias* bias)
{
  this->biases.push_back(bias);
}


void RecurrentNeuralNetwork::setInput(Eigen::VectorXd input)
{
  this->inputLayer->setInput(input);
  this->inputLayer->activate();
}


void RecurrentNeuralNetwork::setTarget(Eigen::VectorXd target)
{
  this->targetLayer->setInput(target);
  this->targetLayer->activate();
}


void RecurrentNeuralNetwork::setTargetLayer(Layer* layer)
{
  this->targetLayer = layer;
}


void RecurrentNeuralNetwork::setObjectiveLayer(ObjectiveLayer* layer)
{
  this->objectiveLayer = layer;
}


void RecurrentNeuralNetwork::forward()
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

void RecurrentNeuralNetwork::backward()
{
  // Start backpropping at the objective layer.  
  for(int i=this->layers.size() - 1; i>0; i--)
  {
    this->layers.at(i)->backprop();
  }
}


vector<Connection*> RecurrentNeuralNetwork::getConnections()
{
  return this->connections;
}


vector<Bias*> RecurrentNeuralNetwork::getBiases()
{
  return this->biases;
}

vector<Layer*> RecurrentNeuralNetwork::getLayers()
{
  return this->layers;
}


Layer* RecurrentNeuralNetwork::getOutputLayer()
{
  return this->outputLayer;
}


ObjectiveLayer* RecurrentNeuralNetwork::getObjectiveLayer()
{
  return this->objectiveLayer;
}


Layer* RecurrentNeuralNetwork::getInputLayer()
{
  return this->inputLayer;
}
