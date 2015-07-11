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
#include <Eigen/Dense>

#include "Layer.h"
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
  vector<vector<Eigen::VectorXd> > delta_history;
  vector<vector<Eigen::VectorXd> > activation_history;
  vector<vector<Eigen::VectorXd> > net_input_history;

  // Initialize the gradients to zero
  for(int i=0; i<this->connections.size(); i++)
  {
    gradients.push_back(0.0*this->connections.at(i)->getGradient());
  }

  // First, reset all the recurrent layers
  for(int i=0; i<this->recurrentLayers.size(); i++)
  {
    recurrentLayers.at(i)->setRecurrentInput(0*recurrentLayers.at(i)->getRecurrentInput());
  }

  // Perform a forward pass to get activations at each time step.

  // Loop through the data, performing each activation
  while(sequence->hasNext())
  {
    vector<Eigen::VectorXd> activations;
    vector<Eigen::VectorXd> net_inputs;
   
    SupervisedData data = sequence->next();
    this->setInput(data.getInput());
    this->setTarget(data.getTarget());

    this->forward();

    for(int i=0; i<this->layers.size(); i++)
    {
      net_inputs.push_back(layers.at(i)->getInput());
      activations.push_back(layers.at(i)->getOutput());
    }

    // Step each recurrent layer to advance to the next time step
    for(int i=0; i<this->recurrentLayers.size(); i++)
    {
      this->recurrentLayers.at(i)->step();
    }

    net_input_history.push_back(net_inputs);
    activation_history.push_back(activations);
  } 

  // What are the activations?
  /*
  for(int i=0; i<activation_history.size(); i++)
  {
    cout << "t = " << i << endl;
    for(int j=0; j<activation_history.at(i).size(); j++)
    {
      cout << "  " << this->layers.at(j)->getName() << ": ";
      cout << activation_history.at(i).at(j).transpose() << endl;
    }
  }
  */
 
  // Backpropagate the errors.  The final deltas need to be set to zero,
  // so that no weight update is propagated backwards through time.
  for(int i=0; i<this->recurrentLayers.size(); i++)
  {
    Layer* recurrentLayer = this->recurrentLayers.at(i)->getRecurrentConnection();
    Eigen::VectorXd delta = recurrentLayer->getDeltas();
    recurrentLayer->clearDeltas();
  }

  // Starting with the final activation, work backwards through time--
  // Calculate the deltas, get the gradients, then step the delta's 
  // backward through time.
  for(int i=activation_history.size()-1; i>=0; i--)
  {
    vector<Eigen::VectorXd> activations = activation_history.at(i);
    vector<Eigen::VectorXd> net_inputs = net_input_history.at(i);

    // Set the activations for this time step
    for(int j=0; j<this->layers.size(); j++)
    {
      layers.at(j)->setInput(activations.at(j));
      layers.at(j)->setOutput(activations.at(j));
    }

    // Set the target output
    this->setInput(sequence->getDataAt(i).getInput());
    this->setTarget(sequence->getDataAt(i).getTarget());
    this->inputLayer->activate();
//    this->targetLayer->activate();

    // And backprop
    this->backward();

    // Update the gradients
    for(int j=0; j<this->connections.size(); j++)
    {
      gradients.at(j) += this->connections.at(j)->getGradient();
    }

    // Move the deltas backwards in the recurrent layers
    for(int j=0; j<this->recurrentLayers.size(); j++)
    {
      this->recurrentLayers.at(j)->backstep();
    }
  }

  // The gradients are averaged over the total time
  for(int i=0; i<gradients.size(); i++)
  {
    gradients.at(i) /= sequence->getLength();
  }

  return gradients;
}
