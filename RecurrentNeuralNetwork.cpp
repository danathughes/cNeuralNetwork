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
#include <typeinfo>

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

  // Is this a recurrent layer?
  if(typeid(*layer) == typeid(RecurrentLayer))
  {
    // Downcast it as such and add to the list of recurrentLayers
    RecurrentLayer* recurrentLayer = dynamic_cast<RecurrentLayer*>(layer);
    this->recurrentLayers.push_back(recurrentLayer);
  }
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
  this->connections.push_back((Connection*) bias);
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

/*
vector<Bias*> RecurrentNeuralNetwork::getBiases()
{
  return this->biases;
}
*/

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


vector<Eigen::MatrixXd> RecurrentNeuralNetwork::getParameterGradients(Sequence* sequence)
{
  vector<Eigen::MatrixXd> gradients;
  
  // First, reset all the recurrent layers
  for(int i=0; i<this->recurrentLayers.size(); i++)
  {
    recurrentLayers.at(i)->setRecurrentInput(0*recurrentLayers.at(i)->getRecurrentInput());
  }

  // Perform a forward pass to get activations at each time step.
  vector<vector<Eigen::VectorXd> > activation_history;

  // Loop through the data, performing each activation
  while(sequence->hasNext())
  {
    vector<Eigen::VectorXd> activations;
    SupervisedData data = sequence->next();
    this->setInput(data.getInput());
    this->setTarget(data.getTarget());

    this->forward();

    for(int i=0; i<this->layers.size(); i++)
    {
      activations.push_back(layers.at(i)->getOutput());
    }

    // Step each recurrent layer to advance to the next time step
    for(int i=0; i<this->recurrentLayers.size(); i++)
    {
      this->recurrentLayers.at(i)->step();
    }

    activation_history.push_back(activations);
  } 

  // What are the activations?
  for(int i=0; i<activation_history.size(); i++)
  {
    cout << "t = " << i << endl;
    for(int j=0; j<activation_history.at(i).size(); j++)
    {
      cout << "  " << activation_history.at(i).at(j).transpose() << endl;
    }
  }

  // Backpropagate the errors
//  this->backward();

  // Get the gradients in each layer
/*
  for(int i=0; i<this->connections.size(); i++)
  {
    gradients.push_back(this->connections.at(i)->getGradient());
  }
*/

  return gradients;
}
