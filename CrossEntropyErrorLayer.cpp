/**
* \class CrossEntropyErrorLayer
*
* \brief Not sure if this is what I want to do, but let's call this the
*        neural network super duper class for now.
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

#include <Eigen/Dense>

#include "Layer.h"
#include "CrossEntropyErrorLayer.h"

using namespace std;

CrossEntropyErrorLayer::CrossEntropyErrorLayer(int size, Layer* target) : ObjectiveLayer(size)
{
  this->target = target;
}


CrossEntropyErrorLayer::~CrossEntropyErrorLayer()
{

}


void CrossEntropyErrorLayer::activate()
{
  for(int i=0; i<this->getSize(); i++)
  {
    this->activations(i) = -(this->target->getOutput()(i) * log(this->net_input(i))
    this->activations(i) -= (1.0 - this->target->getOutput()(i) * log(1.0 - this->net_input(i))
  }
}


Eigen::VectorXd CrossEntropyErrorLayer::gradient()
{
\  return (this->target->getOutput() - this->net_input) / (this->net_input * (1.0 - this->net_input));
}


double CrossEntropyErrorLayer::cost()
{
  return this->activations.sum();
}


void CrossEntropyErrorLayer::backprop()
{
  Eigen::VectorXd grad = this->gradient();
  for(int i=0; i<this->getSize(); i++)
  {
    this->deltas(i) = grad(i);
  }
}
