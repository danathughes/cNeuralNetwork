/**
*
*/

#include "Layer.h"
#include <Eigen/Dense>
#include "Bias.h"
#include "Connection.h"
#include "string"

using namespace std;

Layer::Layer(int size)
{
  this->size = size;
  this->net_input = Eigen::VectorXd(size);
  this->activations = Eigen::VectorXd(size);
  this->deltas = Eigen::VectorXd(size);
}


Layer::~Layer()
{

}


void Layer::setName(string name)
{
  this->name = name;
}

string Layer::getName()
{
  return this->name;
}


int Layer::getSize()
{
  return this->size;
}


void Layer::setInput(Eigen::VectorXd input)
{
  for(int i=0; i<this->size; i++)
  {
    this->net_input(i) = input(i);
  }
}


void Layer::setOutput(Eigen::VectorXd output)
{
  for(int i=0; i<this->size; i++)
  {
    this->activations(i) = output(i);
  }
}


Eigen::VectorXd Layer::getInput()
{
  return this->net_input;
}


Eigen::VectorXd Layer::getOutput()
{
  return this->activations;
}


Eigen::VectorXd Layer::getDeltas()
{
  return this->deltas;
}


void Layer::setDelta(Eigen::VectorXd delta)
{
  for(int i=0; i<this->size; i++)
  {
    this->deltas(i) = delta(i);
  }
}


void Layer::clearDeltas()
{
  this->deltas *= 0.0;
}


void Layer::addInputConnection(Connection* connection)
{
  this->inputConnections.push_back(connection);
}


void Layer::addOutputConnection(Connection* connection)
{
  this->outputConnections.push_back(connection);
}


void Layer::addBias(Bias* bias)
{
  this->biases.push_back(bias);
}


vector<Connection*> Layer::getInputConnections()
{
  return this->inputConnections;
}


vector<Bias*> Layer::getBiases()
{
  return this->biases;
}


void Layer::calculateNetInput()
{
  // Initialize the current net input to zero
  this->net_input *= 0.0;

  // Calculate the contributions from prior layers
  for(int i=0; i<this->inputConnections.size(); i++)
  {
    this->net_input += this->inputConnections.at(i)->getNetOutput();   
  }

  // Add the biases
  for(int i=0; i<this->biases.size(); i++)
  {
    this->net_input += this->biases.at(i)->getBias();
  }
}


void Layer::backprop()
{
  // Clear out the deltas
  this->deltas *= 0.0;

  // Get the gradient of the activations
  Eigen::VectorXd grad = this->gradient();

  // And the backpropagated delta from the other layers
  for(int i=0; i<this->outputConnections.size(); i++)
  {
    this->deltas += this->outputConnections.at(i)->backpropDelta();
  } 
  for(int i=0; i<this->getSize(); i++)
  {
    this->deltas(i) *= grad(i);
  }
}
