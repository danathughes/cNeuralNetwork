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
    this->activations(i) = -(this->target->getOutput()(i) * log(this->net_input(i)));
    this->activations(i) -= (1.0 - this->target->getOutput()(i) * log(1.0 - this->net_input(i)));
  }
}


Eigen::VectorXd CrossEntropyErrorLayer::gradient()
{
  Eigen::VectorXd grad = Eigen::VectorXd(this->getSize());

  for(int i=0; i<this->getSize(); i++)
  {
    grad(i) = (this->target->getOutput()(i) - this->getInput()(i));
    grad(i) /= (this->getInput()(i) * (1.0 - this->getInput()(i) ));
  }
  return grad;
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
