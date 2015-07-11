/**
*
*/

#include "RecurrentLayer.h"
#include "LinearLayer.h"

#include <Eigen/Dense>
#include <math.h>

using namespace std;

RecurrentLayer::RecurrentLayer(Layer* layer) : Layer(layer->getSize())
{
  this->mainLayer = layer;
  this->recurrentLayer = new LinearLayer(layer->getSize());
}

RecurrentLayer::~RecurrentLayer()
{

}

void RecurrentLayer::activate()
{
  mainLayer->setInput(this->net_input + this->recurrentLayer->getOutput());
  mainLayer->activate();
  this->activations = mainLayer->getOutput();
}


Eigen::VectorXd RecurrentLayer::gradient()
{
  return this->mainLayer->gradient();
}


void RecurrentLayer::setRecurrentInput(Eigen::VectorXd input)
{
  this->recurrentLayer->setInput(input);
  this->recurrentLayer->activate();
}

Eigen::VectorXd RecurrentLayer::getRecurrentInput()
{
  return this->recurrentLayer->getInput();
}

void RecurrentLayer::step()
{
  // Step simply calculates the net input of the recurrent layer and activates it.
  this->recurrentLayer->calculateNetInput();
  this->recurrentLayer->activate();
}


void RecurrentLayer::backstep()
{
  this->recurrentLayer->setDelta(this->deltas);
}


Layer* RecurrentLayer::getRecurrentConnection()
{
  return recurrentLayer;
}
